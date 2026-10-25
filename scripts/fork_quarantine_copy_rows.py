#!/usr/bin/env python3
"""fork: list / quarantine re-stored copies of already stored rows that wedge compaction.

Dry-run by default. ``--apply`` needs a backup made first (``sqlite3 lcm.db ".backup <file>"``) and a session with
no turn in flight. Nothing is deleted or truncated; see ``fork_copy_quarantine.py``. Undo with
``scripts/fork_quarantine_summary_rows.py --db ... --restore <ids>`` (same table).

  python3 scripts/fork_quarantine_copy_rows.py --db ~/.hermes/lcm.db --state-db ~/.hermes/state.db --session <id>
  python3 scripts/fork_quarantine_copy_rows.py --db ... --state-db ... --session <id> --apply --backup-confirmed <path>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
_spec = importlib.util.spec_from_file_location("fork_copy_quarantine", _HERE / "fork_copy_quarantine.py")
fcq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fcq)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--state-db", default="")
    ap.add_argument("--session", action="append", default=[], required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--backup-confirmed", default="")
    args = ap.parse_args(argv)
    db = os.path.expanduser(args.db)
    state_db = os.path.expanduser(args.state_db) if args.state_db else None
    if args.apply and not (args.backup_confirmed and Path(os.path.expanduser(args.backup_confirmed)).is_file()):
        print("refusing --apply: pass --backup-confirmed <existing backup file>", file=sys.stderr)
        return 2
    uri = Path(db).resolve().as_uri() + ("" if args.apply else "?mode=ro")
    conn = sqlite3.connect(uri, uri=True, timeout=30, isolation_level=None)
    out = []
    for session_id in args.session:
        result = (fcq.apply(conn, session_id, state_db=state_db) if args.apply
                  else fcq.plan(conn, session_id, state_db=state_db))
        roles = Counter(conn.execute("SELECT role FROM messages WHERE store_id = ?", (s,)).fetchone()[0]
                        for s, _e, _c in result["rows"])
        out.append({"session_id": session_id, "applied": bool(args.apply and result["rows"]),
                    "copies": len(result["rows"]), "by_role": dict(roles),
                    "first": [{"store_id": s, "copy_of": e} for s, e, _c in result["rows"][:5]],
                    "frontier_moves": [dict(zip(("conversation_id", "field", "from", "to"), m))
                                       for m in result["frontier_moves"]]})
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
