#!/usr/bin/env python3
"""fork: list / quarantine / restore raw copies of LCM's own summary rows that wedge compaction.

Dry-run by default. ``--apply`` needs a backup made first (``sqlite3 lcm.db ".backup <file>"``) and a session with no
turn in flight. Nothing is deleted or truncated; see ``fork_summary_quarantine.py``.

  python3 scripts/fork_quarantine_summary_rows.py --db ~/.hermes/profiles/sion/lcm.db --session <id>
  python3 scripts/fork_quarantine_summary_rows.py --db ... --session <id> --apply --backup-confirmed <path>
  python3 scripts/fork_quarantine_summary_rows.py --db ... --restore 163884,164107
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("fork_summary_quarantine", _HERE / "fork_summary_quarantine.py")
fsq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fsq)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--session", action="append", default=[])
    ap.add_argument("--scope", choices=("frontier", "all"), default="frontier")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--backup-confirmed", default="")
    ap.add_argument("--restore", default="")
    args = ap.parse_args(argv)
    db = os.path.expanduser(args.db)
    if args.restore:
        conn = sqlite3.connect(db, timeout=30, isolation_level=None)
        print(json.dumps({"restored": fsq.restore(conn, [int(x) for x in args.restore.split(",") if x.strip()])}))
        return 0
    if args.apply and not (args.backup_confirmed and Path(os.path.expanduser(args.backup_confirmed)).is_file()):
        print("refusing --apply: pass --backup-confirmed <existing backup file>", file=sys.stderr)
        return 2
    uri = Path(db).resolve().as_uri() + ("" if args.apply else "?mode=ro")
    conn = sqlite3.connect(uri, uri=True, timeout=30, isolation_level=None)
    out = []
    for session_id in args.session:
        result = (fsq.apply(conn, session_id, reason="cold-bind summary head stored as raw row", scope=args.scope)
                  if args.apply else fsq.plan(conn, session_id, scope=args.scope))
        out.append({"session_id": session_id, "applied": bool(args.apply and result["rows"]),
                    "rows": [{"store_id": s, "nodes": n, "chars": c} for s, n, c in result["rows"]],
                    "frontier_moves": [dict(zip(("conversation_id", "field", "from", "to"), m))
                                       for m in result["frontier_moves"]]})
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
