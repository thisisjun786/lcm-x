"""fork: quarantine raw copies of LCM's own summary rows (lossless, reversible).

A cold bind whose commit proof no longer binds (host move, a changed ``hermes_home``, a pass that left the proof
stale) handed LCM the host list with LCM's own rendered summary head at the top, and ingest stored that head as a
raw user row. That row then sat at or above the compaction frontier. The in-order store mapper tied the host's head
to it, every later host row went unmapped, and each compaction stopped on "leaf would end before an unresolved
retained occurrence (... kind=shown)" or "selected leaf chunk lacks raw store lineage". ``engine.py`` now never
stores such a row (``LCM_FORK_OWN_SUMMARY_GUARD``); this module repairs stores an earlier cold bind already wedged.

A candidate is a ``role='user'`` row of the session whose whole content is one or more summary parts, each byte for
byte the rendering of a DAG node of the same session (``[<label> Summary (dN, node M)]\\n<summary>\\n[Expand for
details: <hint>]``, parts joined by ``\\n\\n---\\n\\n``). Its text is already held by those nodes, so it is never the only
copy of anything. A carrier (summary parts, ``\n\n``, then text byte-equal to an earlier stored user row of the
session) counts too: the nodes hold its head and the earlier row holds its remainder. Nothing is deleted or truncated: the row keeps its store_id, bytes, FTS entry, embeddings and host
uid bindings; only its ``session_id``/``conversation_id`` move to ``<id>::fork-quarantine`` so the session's
selection no longer reads it, and the original values are recorded in ``lcm_fork_quarantined_messages`` (restore with
``restore()``). When the lifecycle frontier points at a quarantined row it moves down to the session's highest
remaining row at or below it (rows below the old frontier stay below it; nothing is uncovered or dropped).
"""

from __future__ import annotations

import re
import sqlite3
import time
from typing import Iterable, Optional

QUARANTINE_SUFFIX = "::fork-quarantine"
TABLE = "lcm_fork_quarantined_messages"
_HEADER_RE = re.compile(r"\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d(\d+), node (\d+)\)\]\n")
_SEP = "\n\n---\n\n"
_LABELS = {0: "Recent", 1: "Session Arc", 2: "Durable"}


def _render(node: sqlite3.Row) -> str:
    depth = int(node["depth"])
    label = _LABELS.get(depth, f"Depth-{depth}")
    return f"[{label} Summary (d{depth}, node {int(node['node_id'])})]\n{node['summary']}\n[Expand for details: {node['expand_hint']}]"


def own_summary_prefix(conn: sqlite3.Connection, session_id: str, content: str) -> tuple[int, list[int]]:
    """(end offset, node ids) of the leading verified summary parts of ``session_id``'s DAG, else (0, [])."""
    if not content.startswith("[") or "[Expand for details:" not in content:
        return 0, []
    pos, nodes = 0, []
    while True:
        header = _HEADER_RE.match(content, pos)
        if header is None:
            return 0, []
        node = conn.execute(
            "SELECT node_id, session_id, depth, summary, expand_hint FROM summary_nodes WHERE node_id = ?",
            (int(header.group(2)),)).fetchone()
        if node is None or node["session_id"] != session_id or int(node["depth"]) != int(header.group(1)):
            return 0, []
        part = _render(node)
        if not content.startswith(part, pos):
            return 0, []
        pos += len(part)
        nodes.append(int(node["node_id"]))
        if content.startswith(_SEP, pos) and _HEADER_RE.match(content, pos + len(_SEP)):
            pos += len(_SEP)
            continue
        return pos, nodes


def is_pure_own_summary(conn: sqlite3.Connection, session_id: str, content: str) -> Optional[list[int]]:
    """Node ids when ``content`` is only verified summary parts of ``session_id``'s DAG, else None."""
    end, nodes = own_summary_prefix(conn, session_id, content)
    return nodes if nodes and not content[end:].strip() else None


def replayed_carrier_twin(conn: sqlite3.Connection, session_id: str, store_id: int, content: str) -> Optional[tuple[list[int], int]]:
    """(node ids, twin store_id) when ``content`` is verified summary parts followed by ``\n\n`` and text that is
    byte for byte (trailing whitespace aside) an EARLIER stored user row of the same session (a host merge of the summary head with a
    replayed old turn), else None. The twin keeps that text, so the carrier holds nothing of its own."""
    end, nodes = own_summary_prefix(conn, session_id, content)
    if not nodes or not content.startswith("\n\n", end) or not content[end + 2:].strip():
        return None
    remainder = content[end + 2:].rstrip()  # Hermes' merge strips the joined row's trailing whitespace
    for twin_id, twin_content in conn.execute(
            "SELECT store_id, content FROM messages WHERE session_id = ? AND role = 'user' AND store_id < ? "
            "AND substr(content, 1, ?) = ? ORDER BY store_id DESC",
            (session_id, int(store_id), len(remainder), remainder)):
        if (twin_content or "").rstrip() == remainder:
            return nodes, int(twin_id)
    return None


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {TABLE} (
            store_id INTEGER PRIMARY KEY,
            original_session_id TEXT NOT NULL,
            original_conversation_id TEXT,
            node_ids TEXT NOT NULL,
            reason TEXT NOT NULL,
            frontier_before INTEGER,
            frontier_after INTEGER,
            quarantined_at REAL NOT NULL,
            restored_at REAL
        )""")


def candidates(conn: sqlite3.Connection, session_id: str, *, min_store_id: int = 0) -> list[tuple[int, list[int], int]]:
    """``[(store_id, node_ids, chars)]`` of the session's raw pure-summary rows (and replayed carriers, whose
    node list ends with ``-<twin store_id>``) above ``min_store_id``."""
    conn.row_factory = sqlite3.Row
    found = []
    for row in conn.execute(
            "SELECT store_id, content FROM messages WHERE session_id = ? AND role = 'user' AND store_id > ? "
            "AND substr(content, 1, 1) = '[' AND instr(content, ' Summary (d') > 0 ORDER BY store_id",
            (session_id, int(min_store_id))):
        content = row["content"] or ""
        nodes = is_pure_own_summary(conn, session_id, content)
        if not nodes:
            carrier = replayed_carrier_twin(conn, session_id, int(row["store_id"]), content)
            nodes = carrier[0] + [-carrier[1]] if carrier else None  # -twin: the remainder's earlier copy
        if nodes:
            found.append((int(row["store_id"]), nodes, len(content)))
    return found


def _lifecycle(conn: sqlite3.Connection, session_id: str):
    return conn.execute(
        "SELECT conversation_id, current_session_id, current_frontier_store_id, last_finalized_session_id, "
        "last_finalized_frontier_store_id FROM lcm_lifecycle_state "
        "WHERE current_session_id = ? OR last_finalized_session_id = ? OR conversation_id = ?",
        (session_id, session_id, session_id)).fetchall()


def plan(conn: sqlite3.Connection, session_id: str, *, scope: str = "frontier") -> dict:
    """What ``apply`` would do. ``scope='frontier'`` (default) takes only rows at or above the lowest lifecycle
    frontier of the session (the rows that can wedge selection); ``scope='all'`` takes every such row."""
    conn.row_factory = sqlite3.Row
    life = _lifecycle(conn, session_id)
    frontiers = [int(r[k] or 0) for r in life for k, owner in (("current_frontier_store_id", "current_session_id"),
                 ("last_finalized_frontier_store_id", "last_finalized_session_id")) if r[owner] == session_id and r[k]]
    floor = 0 if scope == "all" or not frontiers else min(frontiers) - 1
    rows = candidates(conn, session_id, min_store_id=floor)
    ids = {sid for sid, _n, _c in rows}
    moves = []
    for r in life:
        for key, owner in (("current_frontier_store_id", "current_session_id"),
                           ("last_finalized_frontier_store_id", "last_finalized_session_id")):
            old = int(r[key] or 0)
            if r[owner] != session_id or old not in ids:
                continue
            new = conn.execute(
                "SELECT max(store_id) FROM messages WHERE session_id = ? AND store_id <= ? AND store_id NOT IN (%s)"
                % ",".join("?" * len(ids)), (session_id, old, *ids)).fetchone()[0] or 0
            moves.append((r["conversation_id"], key, old, int(new)))
    return {"session_id": session_id, "rows": rows, "frontier_moves": moves}


def apply(conn: sqlite3.Connection, session_id: str, *, reason: str, scope: str = "frontier") -> dict:
    """Quarantine the planned rows and move a frontier that pointed at one, in one transaction."""
    conn.row_factory = sqlite3.Row
    conn.execute("BEGIN IMMEDIATE")
    try:
        result = plan(conn, session_id, scope=scope)
        if not result["rows"]:
            conn.execute("ROLLBACK")
            return result
        _ensure_table(conn)
        before = {m[2] for m in result["frontier_moves"]}
        after = {m[3] for m in result["frontier_moves"]}
        now = time.time()
        for store_id, nodes, _chars in result["rows"]:
            row = conn.execute("SELECT session_id, conversation_id FROM messages WHERE store_id = ?", (store_id,)).fetchone()
            conn.execute(
                f"INSERT OR REPLACE INTO {TABLE} (store_id, original_session_id, original_conversation_id, node_ids, "
                "reason, frontier_before, frontier_after, quarantined_at, restored_at) VALUES (?,?,?,?,?,?,?,?,NULL)",
                (store_id, row["session_id"], row["conversation_id"], ",".join(map(str, nodes)), reason,
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


def restore(conn: sqlite3.Connection, store_ids: Iterable[int]) -> int:
    """Undo ``apply`` for these rows (the frontier is not raised back; a later compaction moves it)."""
    conn.row_factory = sqlite3.Row
    done = 0
    conn.execute("BEGIN IMMEDIATE")
    try:
        for store_id in store_ids:
            q = conn.execute(f"SELECT * FROM {TABLE} WHERE store_id = ? AND restored_at IS NULL", (int(store_id),)).fetchone()
            if q is None:
                continue
            conn.execute("UPDATE messages SET session_id = ?, conversation_id = ? WHERE store_id = ?",
                         (q["original_session_id"], q["original_conversation_id"], int(store_id)))
            conn.execute(f"UPDATE {TABLE} SET restored_at = ? WHERE store_id = ?", (time.time(), int(store_id)))
            done += 1
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return done
