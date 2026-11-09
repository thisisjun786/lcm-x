"""fork: a spent LCM carrier at the head of the host view must not stop compaction (synthetic fixture).

A long Telegram session's host view began with ONE user row the host built itself (role-alternation repair glued LCM's
summary row to the next user row R): verified LCM summary parts of this session, ``\\n\\n``, then R. R had already been
published by a depth-0 node (stored above the frontier, covered sparsely), and one uid the host absorbed into the
carrier is LCM's own earlier summary row (an ``engine`` binding, no stored row), so the absorbed-claim check fails. No
view row maps the carrier (the v4 emission proof keeps its full identity) and it claims nothing, yet it is far over
the chunk budget: ``_uid_leaf_candidates`` handed the leaf builder the carrier ALONE as the oldest chunk, the builder
dropped it (no store id), and every pass ended ``no paired uncovered source rows`` with zero leaves while the
uid-bound raw backlog behind it grew (default telegram session 11-09, frontier stuck, backlog ~420k tokens). The
carrier is now left out of the candidates.
"""

from __future__ import annotations

import itertools

import hermes_lcm.engine as lcm_engine
from hermes_lcm.dag import SummaryNode
from hermes_lcm.tokens import count_messages_tokens
from tests.test_fork_carrier_head_claim import CARRIER_UID, CHUNK, REMAINDER, _engine, _part
from tests.test_fork_covered_head import _rows, _store
from tests.test_fork_rewritten_uid_mapping import bind

SUMMARY_UID = "lcm-engine-summary-0"
SUMMARY_PAD = " the operator reviewed the deployment plan and its rollback notes" * 240


def _spent_carrier_view(tmp_path, monkeypatch):
    """Two real compactions (a v4 emission proof in force), R stored and covered by a depth-0 node ABOVE the
    frontier, then raw rows stored above it. Returns (view, carrier, raw store ids, R's store id)."""
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **kw: ("Earlier turns." + SUMMARY_PAD, 1))
    clock = itertools.count(1_700_000_000)
    old, older = _rows("old", 3, clock), _rows("older", 3, clock)
    engine = _engine(tmp_path, fresh_tail_count=1, chunk=20_000)
    try:
        _store(engine, [*old, older[0]])
        first = [{"role": "system", "content": "sys"}, *[dict(row) for row in old], dict(older[0])]
        engine._ingest_cursor = len(first)
        emitted = engine.compress(first, current_tokens=100_000, force=True)
        _store(engine, older[1:])
        user = {"role": "user", "content": REMAINDER, "timestamp": float(next(clock)), "message_uid": "host-uid-R"}
        r_id = _store(engine, [user])[0]
        second = [*emitted[:-1], *[dict(row) for row in older], dict(user)]
        engine._ingest_cursor = len(second)
        engine.compress(second, current_tokens=100_000, force=True)
        frontier = engine._lifecycle.get_by_conversation("conv").current_frontier_store_id
        # A sparse depth-0 node over R alone, above the frontier: the frontier does not move.
        engine._dag.add_node(SummaryNode(session_id="S", summary="Synthetic summary of R.", depth=0, source_ids=[r_id],
                                         source_type="messages", token_count=8, source_token_count=20))
        raw = _rows("raw", 12, clock)
        raw_ids = _store(engine, raw)
        bind(engine, CARRIER_UID, 0, "engine")
        bind(engine, SUMMARY_UID, 0, "engine")  # the host's own summary row the carrier absorbed: no stored row
        later = max((n for n in engine._dag.get_session_nodes("S") if n.depth == 0 and n.source_ids != [r_id]),
                    key=lambda n: n.node_id)
    finally:
        engine.shutdown()
    assert frontier < r_id < raw_ids[0]
    carrier = {"role": "user", "content": _part(later) + "\n\n" + REMAINDER, "timestamp": user["timestamp"] + 86_400.0,
               "message_uid": CARRIER_UID, "_absorbed_message_uids": [SUMMARY_UID, "host-uid-R"]}
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **kw: ("Synthetic leaf summary.", 1))
    return [{"role": "system", "content": "sys"}, carrier, *[dict(row) for row in raw]], carrier, raw_ids, r_id


def test_spent_carrier_head_does_not_stop_compaction(tmp_path, monkeypatch):
    view, carrier, raw_ids, r_id = _spent_carrier_view(tmp_path, monkeypatch)
    assert count_messages_tokens([carrier]) > CHUNK  # the carrier alone is the oldest leaf chunk
    engine = _engine(tmp_path)  # cold resume: the host replays its whole view
    try:
        stored_before = engine._store.get_session_messages("S")
        nodes_before = {n.node_id for n in engine._dag.get_session_nodes("S")}
        tokens = count_messages_tokens(view)
        result = engine.compress(view)
        reason = engine._last_compression_noop_reason
        new_leaves = [n for n in engine._dag.get_session_nodes("S") if n.depth == 0 and n.node_id not in nodes_before]
        status = engine._uid_coverage_status()
        stored_after = engine._store.get_session_messages("S")
    finally:
        engine.shutdown()
    assert reason != "no paired uncovered source rows", reason
    assert new_leaves, reason
    sources = [sid for node in new_leaves for sid in node.source_ids]
    assert sources and set(sources) <= set(raw_ids)  # raw rows only: R stays covered by its own node, never twice
    assert r_id not in sources
    assert len(result) < len(view) and count_messages_tokens(result) < tokens, (len(result), len(view))
    assert status.get("unpaired_count", 0) == 0, status
    after = {int(row["store_id"]): row for row in stored_after}
    assert all(after.get(int(row["store_id"])) == row for row in stored_before)  # lossless
