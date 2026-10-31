"""fork: a host view whose front is already summarized must not stop compaction (synthetic fixture).

A long session's host view kept, at its front, rows that are already sources of an existing depth-0 summary node
(stored at or below the lifecycle frontier F; the host never marked them compacted), followed by raw rows above F.
Every row carries a bound ``message_uid``. ``_uid_leaf_candidates`` dropped only groups covered ABOVE F, so the oldest
chunk it handed the leaf builder was the covered head; ``_uid_set_leaf_input`` (coverage over every mapped id) then
discarded all of it, and each pass ended ``no paired uncovered source rows`` with zero leaves while the raw backlog
behind the head kept growing. Candidate selection now applies the same covered set as the leaf builder.
"""

from __future__ import annotations

import itertools

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens
from tests.test_fork_late_copy_mapping import _home
from tests.test_fork_rewritten_uid_mapping import bind

PAD = " lorem ipsum dolor sit amet" * 60


def _engine(tmp_path) -> LCMEngine:
    engine = LCMEngine(config=LCMConfig(fresh_tail_count=4, leaf_chunk_tokens=3000, dynamic_leaf_chunk_max=3000,
                                        database_path=str(tmp_path / "lcm.db")), hermes_home=_home(tmp_path))
    engine.on_session_start("S", platform="telegram", context_length=24_000, conversation_id="conv")
    return engine


def _rows(prefix, turns, clock):
    """user turns, each followed by one tool call/result group and a reply; every row has a host uid."""
    rows = []
    for n in range(turns):
        rows.append({"role": "user", "content": f"{prefix} ask {n}" + PAD, "timestamp": float(next(clock))})
        call = {"id": f"{prefix}_c{n}", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}
        rows.append({"role": "assistant", "content": "", "tool_calls": [call], "timestamp": float(next(clock))})
        rows.append({"role": "tool", "content": f"{prefix} output {n}" + PAD, "tool_call_id": call["id"],
                     "timestamp": float(next(clock))})
        rows.append({"role": "assistant", "content": f"{prefix} reply {n}" + PAD, "timestamp": float(next(clock))})
    for i, row in enumerate(rows):
        row["message_uid"] = f"{prefix}-uid-{i}"
    return rows


def _store(engine, rows):
    ids = []
    for row in rows:
        sid = engine._store.append("S", {k: v for k, v in row.items() if k != "message_uid"},
                                   conversation_id="conv")
        bind(engine, row["message_uid"], sid)
        ids.append(sid)
    return ids


def _summarize_head(engine, rows, ids):
    """Publish one depth-0 node over the head rows: the frontier moves to their last store id."""
    engine._uid_source_bindings = {sid: row["message_uid"] for row, sid in zip(rows, ids)}
    engine._store_complete_copy_exclusion_proofs = {}
    frontier = engine._lifecycle.get_by_conversation("conv").current_frontier_store_id
    node = SummaryNode(session_id="S", summary="Synthetic summary of the head.", depth=0, source_ids=list(ids),
                       source_type="messages", token_count=8, source_token_count=5000)
    engine._publish_leaf_node(node, level=1, model="synthetic", expected_frontier=frontier,
                              consumed_store_ids=list(ids), publication_excluded_store_ids=[],
                              filter_exclusion_proofs={}, carried_ranges=[])


def test_summarized_host_head_does_not_stop_compaction(tmp_path, monkeypatch):
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **kw: ("Synthetic leaf summary.", 1))
    clock = itertools.count(1_700_000_000)
    # The host still shows ``head``; ``gone`` is summarized with it but no longer in the view, so the
    # committed-replay drop (#457, an exact run of the rows through F) does not apply, as on the real session.
    head, gone, raw = _rows("head", 6, clock), _rows("gone", 2, clock), _rows("raw", 12, clock)
    engine = _engine(tmp_path)
    try:
        head_ids, gone_ids = _store(engine, head), _store(engine, gone)
        _summarize_head(engine, head + gone, head_ids + gone_ids)
        raw_ids = _store(engine, raw)
        frontier = engine._lifecycle.get_by_conversation("conv").current_frontier_store_id
        assert frontier == gone_ids[-1]
    finally:
        engine.shutdown()
    engine = _engine(tmp_path)  # cold resume: the host replays its whole view, head not marked compacted
    try:
        assert engine._last_compacted_store_id == frontier
        view = [{"role": "system", "content": "sys"}, *[dict(row) for row in head + raw]]
        nodes_before = len(engine._dag.get_session_nodes("S"))
        result = engine.compress(view)
        reason = engine._last_compression_noop_reason
        new_nodes = [n for n in engine._dag.get_session_nodes("S")[nodes_before:] if n.depth == 0]
        sources = {sid for node in new_nodes for sid in node.source_ids}
        status = engine._uid_coverage_status()
        stored = engine._store.get_session_messages("S")
    finally:
        engine.shutdown()
    assert reason != "no paired uncovered source rows", reason
    assert new_nodes, reason
    assert sources and sources <= set(raw_ids)  # leaves cover raw rows only, never re-summarize the head
    assert len(result) < len(view) and count_messages_tokens(result) * 2 < count_messages_tokens(view), (
        len(result), len(view), count_messages_tokens(result), count_messages_tokens(view))
    assert status.get("unpaired_count", 0) == 0, status
    shown = {m.get("message_uid") for m in result}
    assert all(sid in sources or row["message_uid"] in shown for row, sid in zip(raw, raw_ids))
    assert {sid for row in stored for sid in [int(row["store_id"])]} >= set(head_ids + gone_ids + raw_ids)  # lossless
