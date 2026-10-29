#!/usr/bin/env python3
"""Backfill unambiguous host UID bindings. Dry-run by default; stop active turns first.

Apply only after a SQLite backup: --apply --backup-confirmed BACKUP. This script
adds bindings only; it does not change messages. Run separately for each Hermes home.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

# Load the plugin as a package without invoking its registration entry point.
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('hermes_lcm', root / '__init__.py', submodule_search_locations=[str(root)])
pkg = importlib.util.module_from_spec(spec)
sys.modules['hermes_lcm'] = pkg
from hermes_lcm.host_uid import HostUidShadowMixin, _valid_uid  # noqa: E402
from hermes_lcm.reconcile import (  # noqa: E402
    _has_lossy_redacted_identity, _HERMES_PRUNED_TOOL_PLACEHOLDER,
    _is_hermes_persisted_output_marker, _PRESERVED_TODO_CONTEXT_PREFIX,
    _MODEL_SWITCH_NOTIFICATION_PREFIX, _OOB_MESSAGE_BLOCK_MARKER,
)
from hermes_lcm.store import _normalize_observed_at  # noqa: E402


def collision_key(row):
    """Group all plausible collisions, even rows we cannot admit.

    Any two rows with the same admitted identity share a collision key. Ignoring
    calls and tool name deliberately widens the group for malformed or unknown rows.
    """
    return row['role'], row['content'], row['tool_call_id'] or ''


def admit(row):
    """Return a lossless raw identity, or None when equality cannot prove identity."""
    role, content = row['role'], row['content']
    if not isinstance(role, str) or not isinstance(content, str):
        return None
    # Summary carriers start with a bracket; conservatively exclude every
    # bracket-leading payload without requiring a host engine or DAG lookup.
    if (content.startswith('[') or content == _HERMES_PRUNED_TOOL_PLACEHOLDER
            or _is_hermes_persisted_output_marker(content)
            or _PRESERVED_TODO_CONTEXT_PREFIX in content
            or _MODEL_SWITCH_NOTIFICATION_PREFIX in content or _OOB_MESSAGE_BLOCK_MARKER in content):
        return None
    calls = row['tool_calls']
    if calls:
        try:
            parsed = json.loads(calls)
            if not isinstance(parsed, list):
                return None
            calls = json.dumps(parsed, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        except (TypeError, ValueError, RecursionError):
            return None
    else:
        calls = ''
    name = row['tool_name'] if 'tool_name' in row.keys() else None
    if role == 'tool' and (not isinstance(name, str) or not name):
        return None
    if name is not None and not isinstance(name, str):
        return None
    identity = (role, content, row['tool_call_id'] or '', calls, name or '')
    return None if _has_lossy_redacted_identity(identity) else identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True)
    parser.add_argument('--state-db', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--backup-confirmed', default='')
    args = parser.parse_args(argv)
    db, state_path = Path(args.db).expanduser().resolve(), Path(args.state_db).expanduser().resolve()
    backup = Path(args.backup_confirmed).expanduser().resolve() if args.backup_confirmed else None
    if args.apply and (not backup or not backup.is_file() or backup in (db, state_path)):
        parser.error('--apply requires a separate existing --backup-confirmed file')
    state = sqlite3.connect(state_path.as_uri() + '?mode=ro', uri=True)
    conn = sqlite3.connect(db.as_uri() + ('?mode=rw' if args.apply else '?mode=ro'), uri=True, isolation_level=None)
    state.row_factory = conn.row_factory = sqlite3.Row
    try:
        state.execute('BEGIN')
        conn.execute('BEGIN IMMEDIATE' if args.apply else 'BEGIN')
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='host_uid_bindings'").fetchone():
            raise ValueError('host_uid_bindings table is absent; initialize UID bindings before backfill')
        sessions = [r['id'] for r in state.execute('SELECT id FROM sessions')]
        reader = HostUidShadowMixin()
        tag = hashlib.sha256(str(state_path).encode()).hexdigest()[:16]
        groups = defaultdict(list)
        for sid in sessions:
            root_id, _ = reader._host_uid_read_lineage(state_path, str(sid))
            if root_id is not None:
                groups[f'{tag}:{root_id}'].append(str(sid))
        counts = {'timestamp_matched': 0, 'unique_without_timestamp': 0}
        for lineage, ids in groups.items():
            host_groups, stored_groups, uid_counts = defaultdict(list), defaultdict(list), defaultdict(int)
            for sid in ids:
                for row in state.execute('SELECT * FROM messages WHERE session_id = ?', (sid,)):
                    host_groups[collision_key(row)].append(row)
                    if _valid_uid(row['message_uid']):
                        uid_counts[row['message_uid']] += 1
                for row in conn.execute('SELECT * FROM messages WHERE session_id = ?', (sid,)):
                    stored_groups[collision_key(row)].append(row)
            bound_uids = {row['uid'] for row in conn.execute(
                'SELECT uid FROM host_uid_bindings')}
            for key, host_rows in host_groups.items():
                stored_rows = stored_groups.get(key, [])
                if len(host_rows) != 1 or len(stored_rows) != 1:
                    continue
                host, stored = host_rows[0], stored_rows[0]
                identity = admit(host)
                if identity is None or identity != admit(stored):
                    continue
                uid, sid = host['message_uid'], stored['store_id']
                if not _valid_uid(uid) or uid_counts[uid] != 1 or uid in bound_uids:
                    continue
                if conn.execute('SELECT 1 FROM host_uid_bindings WHERE store_id = ? LIMIT 1', (sid,)).fetchone():
                    continue
                observed = stored['observed_at']
                if observed is not None:
                    host_stamp = _normalize_observed_at(host['timestamp']) if 'timestamp' in host.keys() else None
                    if host_stamp is None or host_stamp != _normalize_observed_at(observed):
                        continue
                    path = 'timestamp_matched'
                else:
                    path = 'unique_without_timestamp'
                counts[path] += 1
                if args.apply:
                    conn.execute("INSERT INTO host_uid_bindings (store_id,uid,lineage_key,kind,binding_version,proof_kind,created_at) VALUES (?,?,?,'canonical',1,'backfill',?)", (sid, uid, lineage, time.time()))
        conn.commit()
        total = sum(counts.values())
        print(json.dumps({'candidates': total, 'applied': total if args.apply else 0, **counts}))
    finally:
        conn.close()
        state.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
