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


def payload(row):
    """Conservative exact payload subset: do not reconstruct redacted or rewritten data."""
    if not isinstance(row['role'], str) or not isinstance(row['content'], str):
        return None
    calls = row['tool_calls']
    if calls:
        try:
            calls = json.dumps(json.loads(calls), sort_keys=True, separators=(',', ':'))
        except (TypeError, ValueError):
            return None
    else:
        calls = ''
    return row['role'], row['content'], row['tool_call_id'] or '', calls


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
        total = 0
        for lineage, ids in groups.items():
            host_by_payload, host_by_uid, stored_by_payload = defaultdict(list), defaultdict(list), defaultdict(list)
            for sid in ids:
                for row in state.execute('SELECT * FROM messages WHERE session_id = ?', (sid,)):
                    uid = row['message_uid']
                    key = payload(row)
                    if _valid_uid(uid):
                        host_by_uid[uid].append(row)
                    if key is not None:
                        host_by_payload[key].append(row)
                for row in conn.execute('SELECT * FROM messages WHERE session_id = ?', (sid,)):
                    key = payload(row)
                    if key is not None:
                        stored_by_payload[key].append(row)
            bound_stores, bound_uids = set(), set()
            for binding in conn.execute('SELECT store_id, uid FROM host_uid_bindings WHERE lineage_key = ?', (lineage,)):
                bound_stores.add(binding['store_id'])
                bound_uids.add(binding['uid'])
            for key, host_rows in host_by_payload.items():
                stored_rows = stored_by_payload.get(key, [])
                if len(host_rows) != 1 or len(stored_rows) != 1:
                    continue
                uid, sid = host_rows[0]['message_uid'], stored_rows[0]['store_id']
                if not _valid_uid(uid) or len(host_by_uid[uid]) != 1 or uid in bound_uids or sid in bound_stores:
                    continue
                # A store row associated with another home is not ours to claim.
                if conn.execute('SELECT 1 FROM host_uid_bindings WHERE store_id = ? LIMIT 1', (sid,)).fetchone():
                    continue
                total += 1
                if args.apply:
                    conn.execute("INSERT INTO host_uid_bindings (store_id,uid,lineage_key,kind,binding_version,proof_kind,created_at) VALUES (?,?,?,'canonical',1,'backfill',?)", (sid, uid, lineage, time.time()))
        conn.commit()
        print(json.dumps({'candidates': total, 'applied': total if args.apply else 0}))
    finally:
        conn.close()
        state.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
