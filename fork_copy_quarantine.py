"""fork: quarantine raw RE-STORED copies of rows the session already stored (lossless, reversible).

Hermes' compression re-persists the kept tail with fresh timestamps. Before ``LCM_FORK_UID_REPLAY_STAMPED`` the
first ingest after each compression stored that whole tail again (the default telegram session
20261024_165049_398f126f held 20,755 rows of which 2,925 were distinct on 10-26). Those copies took the in-order
store mapper's slots, so leaves stopped on "unresolved retained occurrence ... kind=shown" or "cannot form a
contiguous stored prefix". The ingest switch stops new copies; this module takes the old ones out of selection.

A row is a copy only with BOTH proofs:
- identity: the row carries the same host ``message_uid`` as an EARLIER stored row of the session, by a canonical or
  version host-uid binding (any home tag of the session root) or by the host transcript (state.db: same uid, same
  role, the row's ``observed_at`` equal to that uid's host stamp);
- bytes: role, content, tool_calls and tool_call_id equal that earlier row's (or the row is the host's
  ``[Old tool output cleared ...]`` placeholder of the same tool_call_id while the earlier row holds the output).

The earliest stored row of each uid is never touched. Nothing is deleted or truncated: as
``fork_summary_quarantine``, only ``session_id``/``conversation_id`` get ``::fork-quarantine``, the originals go to
``lcm_fork_quarantined_messages`` (node_ids = ``copy-of:<earlier store_id>``), and ``restore()`` there undoes it.
A frontier pointing at a quarantined row moves down to the highest remaining row at or below it.
"""

from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

try:  # loaded as hermes_lcm.<module> by the plugin and tests, or standalone by scripts/
    from .fork_summary_quarantine import QUARANTINE_SUFFIX, TABLE, _ensure_table, _lifecycle
except ImportError:
    from fork_summary_quarantine import QUARANTINE_SUFFIX, TABLE, _ensure_table, _lifecycle

PLACEHOLDER = "[Old tool output cleared to save context space]"


def _host_uids(state_db: Optional[str], session_id: str) -> dict:
    """(role, rounded stamp) -> [(uid, content, tool_call_id)] from the host transcript, read-only."""
    out: dict = defaultdict(list)
    if not state_db or not Path(state_db).is_file():
        return out
    st = sqlite3.connect(Path(state_db).resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    try:
        for uid, ts, role, content, tcid in st.execute(
                "SELECT message_uid, timestamp, role, coalesce(content,''), coalesce(tool_call_id,'') FROM messages "
                "WHERE session_id = ? AND message_uid IS NOT NULL AND timestamp IS NOT NULL", (session_id,)):
            out[(role, round(float(ts), 4))].append((uid, content, tcid))
    finally:
        st.close()
    return out


def plan(conn: sqlite3.Connection, session_id: str, *, state_db: Optional[str] = None) -> dict:
    conn.row_factory = sqlite3.Row
    rows = {int(r["store_id"]): r for r in conn.execute(
        "SELECT store_id, role, coalesce(content,'') AS content, coalesce(tool_calls,'') AS tool_calls, "
        "coalesce(tool_call_id,'') AS tool_call_id, observed_at FROM messages WHERE session_id = ?", (session_id,))}
    uids_of: dict = defaultdict(set)
    for uid, store_id in conn.execute(
            "SELECT uid, store_id FROM host_uid_bindings WHERE kind IN ('canonical','version') AND store_id > 0 "
            "AND lineage_key LIKE ?", ("%:" + session_id,)):
        if int(store_id) in rows:
            uids_of[int(store_id)].add(str(uid))
    host = _host_uids(state_db, session_id)
    for store_id, r in rows.items():
        if r["observed_at"] is None:
            continue
        for uid, content, tcid in host.get((r["role"], round(float(r["observed_at"]), 4)), ()):
            if content == r["content"] and tcid == r["tool_call_id"]:
                uids_of[store_id].add(str(uid))
    first: dict = {}
    for store_id in sorted(rows):
        for uid in uids_of.get(store_id, ()):
            first.setdefault(uid, store_id)
    found = []
    for store_id in sorted(rows):
        r = rows[store_id]
        for uid in sorted(uids_of.get(store_id, ())):
            earlier = first[uid]
            if earlier == store_id:
                continue
            a = rows[earlier]
            same = (r["role"], r["content"], r["tool_calls"], r["tool_call_id"]) == (
                a["role"], a["content"], a["tool_calls"], a["tool_call_id"])
            cleared = (r["role"] == "tool" and r["content"] == PLACEHOLDER and r["tool_call_id"]
                       and r["tool_call_id"] == a["tool_call_id"] and len(a["content"]) > len(PLACEHOLDER))
            if same or cleared:
                found.append((store_id, earlier, len(r["content"])))
                break
    ids = {s for s, _e, _c in found}
    moves = []
    for life in _lifecycle(conn, session_id):
        for key, owner in (("current_frontier_store_id", "current_session_id"),
                           ("last_finalized_frontier_store_id", "last_finalized_session_id")):
            old = int(life[key] or 0)
            if life[owner] != session_id or old not in ids:
                continue
            new = max((s for s in rows if s <= old and s not in ids), default=0)
            moves.append((life["conversation_id"], key, old, int(new)))
    return {"session_id": session_id, "rows": found, "frontier_moves": moves}


def apply(conn: sqlite3.Connection, session_id: str, *, state_db: Optional[str] = None,
          reason: str = "restamped re-store of an already stored row") -> dict:
    conn.row_factory = sqlite3.Row
    conn.execute("BEGIN IMMEDIATE")
    try:
        result = plan(conn, session_id, state_db=state_db)
        if not result["rows"]:
            conn.execute("ROLLBACK")
            return result
        _ensure_table(conn)
        before = {m[2] for m in result["frontier_moves"]}
        after = {m[3] for m in result["frontier_moves"]}
        now = time.time()
        for store_id, earlier, _chars in result["rows"]:
            row = conn.execute("SELECT session_id, conversation_id FROM messages WHERE store_id = ?", (store_id,)).fetchone()
            conn.execute(
                f"INSERT OR REPLACE INTO {TABLE} (store_id, original_session_id, original_conversation_id, node_ids, "
                "reason, frontier_before, frontier_after, quarantined_at, restored_at) VALUES (?,?,?,?,?,?,?,?,NULL)",
                (store_id, row["session_id"], row["conversation_id"], f"copy-of:{earlier}", reason,
                 max(before) if before else None, min(after) if after else None, now))
            conn.execute(
                "UPDATE messages SET session_id = session_id || ?, conversation_id = COALESCE(conversation_id, '') || ? "
                "WHERE store_id = ? AND session_id = ?", (QUARANTINE_SUFFIX, QUARANTINE_SUFFIX, store_id, session_id))
        for conversation_id, key, old, new in result["frontier_moves"]:
            conn.execute(f"UPDATE lcm_lifecycle_state SET {key} = ?, updated_at = ? WHERE conversation_id = ? AND {key} = ?",
                         (new, now, conversation_id, old))
        conn.execute("COMMIT")
        return result
    except Exception:
        conn.execute("ROLLBACK")
        raise
