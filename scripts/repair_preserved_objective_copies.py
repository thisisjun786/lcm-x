#!/usr/bin/env python3
"""fork: mark older stored copies of a preserved-objective row superseded (one-shot, lossless, reversible).

Before the preserved-objective fix, a host that merged its next user row onto LCM's objective row made LCM store
an objective row again at many compactions: raw user rows that begin with
"[Current user objective preserved from compacted history]", several of them byte-identical. They had no host uid
binding, stayed unpaired and held the compaction frontier.

Per session, every raw user row with that prefix is grouped by its exact bytes; in each group the newest row stays
live and every older one is marked superseded. Rows with distinct bytes are all kept live (each may hold a merged
user line). Marking reuses the fork quarantine of ``fork_summary_quarantine.py``: the row keeps its store_id, bytes,
FTS entry, embeddings and bindings; only ``session_id``/``conversation_id`` get ``::fork-quarantine`` so selection no
longer reads it, and the originals are recorded in ``lcm_fork_quarantined_messages`` (``node_ids`` =
``superseded-by:<newest store_id>``). Nothing is deleted or truncated; ``--restore`` undoes it. A frontier that points
at a marked row moves down to the highest remaining row at or below it.

Dry run by default (read-only, prints what would change). ``--apply`` first writes a SQLite-safe backup to
``--backup PATH`` (which must not exist yet), then marks the rows in one transaction. Stop the gateway first.

  python3 scripts/repair_preserved_objective_copies.py --db <lcm.db>
  python3 scripts/repair_preserved_objective_copies.py --db <lcm.db> --session <id> --apply --backup <new file>
  python3 scripts/repair_preserved_objective_copies.py --db <lcm.db> --restore 101,102
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("fork_summary_quarantine", _HERE / "fork_summary_quarantine.py")
fsq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fsq)

PREFIX = "[Current user objective preserved from compacted history]"
REASON = "superseded preserved-objective copy"


def plan(conn: sqlite3.Connection, session_id: str) -> dict:
    """``{"session_id", "copies", "rows": [(store_id, newest_id, chars)], "frontier_moves"}`` (read-only)."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT store_id, content FROM messages WHERE session_id = ? AND role = 'user' AND substr(content, 1, ?) = ? "
        "ORDER BY store_id", (session_id, len(PREFIX), PREFIX)).fetchall()
    newest: dict = {}
    for row in rows:
        newest[row["content"]] = int(row["store_id"])
    marked = [(int(row["store_id"]), newest[row["content"]], len(row["content"])) for row in rows
              if int(row["store_id"]) != newest[row["content"]]]
    ids = {store_id for store_id, _newest, _chars in marked}
    moves = []
    for life in fsq._lifecycle(conn, session_id):
        for key, owner in (("current_frontier_store_id", "current_session_id"),
                           ("last_finalized_frontier_store_id", "last_finalized_session_id")):
            old = int(life[key] or 0)
            if life[owner] != session_id or old not in ids:
                continue
            new = conn.execute(
                "SELECT max(store_id) FROM messages WHERE session_id = ? AND store_id <= ? AND store_id NOT IN (%s)"
                % ",".join("?" * len(ids)), (session_id, old, *ids)).fetchone()[0] or 0
            moves.append((life["conversation_id"], key, old, int(new)))
    return {"session_id": session_id, "copies": len(rows), "rows": marked, "frontier_moves": moves}


def apply(conn: sqlite3.Connection, session_id: str) -> dict:
    conn.row_factory = sqlite3.Row
    conn.execute("BEGIN IMMEDIATE")
    try:
        result = plan(conn, session_id)
        if not result["rows"]:
            conn.execute("ROLLBACK")
            return result
        fsq._ensure_table(conn)
        before = {move[2] for move in result["frontier_moves"]}
        after = {move[3] for move in result["frontier_moves"]}
        now = time.time()
        for store_id, newest_id, _chars in result["rows"]:
            row = conn.execute("SELECT session_id, conversation_id FROM messages WHERE store_id = ?", (store_id,)).fetchone()
            conn.execute(
                f"INSERT OR REPLACE INTO {fsq.TABLE} (store_id, original_session_id, original_conversation_id, node_ids, "
                "reason, frontier_before, frontier_after, quarantined_at, restored_at) VALUES (?,?,?,?,?,?,?,?,NULL)",
                (store_id, row["session_id"], row["conversation_id"], f"superseded-by:{newest_id}", REASON,
                 max(before) if before else None, min(after) if after else None, now))
            conn.execute(
                "UPDATE messages SET session_id = session_id || ?, conversation_id = COALESCE(conversation_id, '') || ? "
                "WHERE store_id = ? AND session_id = ?",
                (fsq.QUARANTINE_SUFFIX, fsq.QUARANTINE_SUFFIX, store_id, session_id))
        for conversation_id, key, old, new in result["frontier_moves"]:
            conn.execute(f"UPDATE lcm_lifecycle_state SET {key} = ?, updated_at = ? WHERE conversation_id = ? AND {key} = ?",
                         (new, now, conversation_id, old))
        conn.execute("COMMIT")
        return result
    except Exception:
        conn.execute("ROLLBACK")
        raise


def sessions_with_copies(conn: sqlite3.Connection) -> list[str]:
    return [str(row[0]) for row in conn.execute(
        "SELECT DISTINCT session_id FROM messages WHERE role = 'user' AND substr(content, 1, ?) = ? "
        "AND session_id NOT LIKE ? ORDER BY session_id", (len(PREFIX), PREFIX, "%" + fsq.QUARANTINE_SUFFIX))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--session", action="append", default=[], help="session_id (repeatable; default: every session)")
    ap.add_argument("--apply", action="store_true", help="mark the rows (default: dry run)")
    ap.add_argument("--backup", type=Path, help="backup file written before --apply; must not exist")
    ap.add_argument("--restore", default="", help="comma-separated store_ids to restore")
    args = ap.parse_args(argv)
    db = args.db.expanduser()
    if not db.is_file():
        ap.error(f"no such store: {db}")
    if args.restore:
        conn = sqlite3.connect(str(db), timeout=30, isolation_level=None)
        try:
            print(json.dumps({"restored": fsq.restore(conn, [int(x) for x in args.restore.split(",") if x.strip()])}))
        finally:
            conn.close()
        return 0
    if args.apply and (args.backup is None or args.backup.expanduser().exists()):
        ap.error("--apply needs --backup PATH for a file that does not exist yet")
    uri = db.resolve().as_uri() + ("" if args.apply else "?mode=ro")
    conn = sqlite3.connect(uri, uri=True, timeout=30, isolation_level=None)
    try:
        sessions = args.session or sessions_with_copies(conn)
        if args.apply:
            target = sqlite3.connect(str(args.backup.expanduser()))
            try:
                conn.backup(target)
            finally:
                target.close()
            print(f"backup written: {args.backup}")
        messages_before = conn.execute("SELECT count(*) FROM messages").fetchone()[0]
        out, total = [], 0
        for session_id in sessions:
            result = apply(conn, session_id) if args.apply else plan(conn, session_id)
            total += len(result["rows"])
            out.append({"session_id": session_id, "objective_copies": result["copies"],
                        "applied": bool(args.apply and result["rows"]),
                        ("superseded" if args.apply else "would_supersede"): [
                            {"store_id": s, "superseded_by": n, "chars": c} for s, n, c in result["rows"]],
                        "frontier_moves": [dict(zip(("conversation_id", "field", "from", "to"), m))
                                           for m in result["frontier_moves"]]})
        print(json.dumps(out, ensure_ascii=False, indent=1))
        messages_after = conn.execute("SELECT count(*) FROM messages").fetchone()[0]
        if messages_after != messages_before:  # pragma: no cover - the script never inserts or deletes messages
            print("message count changed; restore the backup", file=sys.stderr)
            return 2
        print(f"{'applied' if args.apply else 'dry run'}: {total} row(s) "
              f"{'marked' if args.apply else 'would be marked'} superseded; messages unchanged ({messages_after})")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
