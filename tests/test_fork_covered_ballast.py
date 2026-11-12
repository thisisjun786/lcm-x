"""fork: host groups that existing leaf nodes already cover are dropped from the host view (synthetic fixture).

A long session's host view kept a large block of rows that are all sources of existing depth-0 nodes of the session
(a compaction run publishes each leaf as it goes and rewrites the host view only at its end: a process killed between
the two leaves the committed sources in the view). One old store row just above the lifecycle frontier F is uncovered,
unbound and not shown (an orphan), so F cannot move and the #457 committed-replay drop (an exact run of the rows
through F) never applies. ``_uid_leaf_candidates`` only skipped the covered groups, so every pass kept them in the
returned context next to the summaries that represent them. The first leaf pass of a compaction now drops every host
group whose rows all map to store rows this session's nodes already cover; anything uncovered, unmapped or in the
fresh tail stays.
"""

from __future__ import annotations

import itertools
import json

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm import tokens as lcm_tokens
from hermes_lcm.dag import SummaryNode
from hermes_lcm.tokens import count_messages_tokens
from tests.test_fork_covered_head import _engine, _rows, _store

COVERED_SUMMARY = "Synthetic summary of the covered block."
OLD_SUMMARY = "Synthetic summary of the oldest turns."
LEAF_SUMMARY = "Synthetic leaf summary."
UNBOUND_UID = "host-uid-never-bound"


@pytest.fixture(autouse=True)
def character_estimate(monkeypatch):
    monkeypatch.setattr(lcm_tokens, "_get_encoder", lambda: None)
    monkeypatch.setattr(lcm_tokens, "_encoder_generation", -1)


@pytest.fixture(autouse=True)
def leaf_summary(monkeypatch):
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **kw: (LEAF_SUMMARY, 1))


def _frontier(engine) -> int:
    return engine._lifecycle.get_by_conversation("conv").current_frontier_store_id


def _publish(engine, rows, ids, summary):
    """One depth-0 node over ``ids``, published as a compaction run publishes it (sparse uid coverage)."""
    engine._uid_source_bindings = {sid: row["message_uid"] for row, sid in zip(rows, ids)}
    engine._store_complete_copy_exclusion_proofs = {}
    node = SummaryNode(session_id="S", summary=summary, depth=0, source_ids=list(ids), source_type="messages",
                       token_count=8, source_token_count=5000)
    engine._publish_leaf_node(node, level=1, model="synthetic", expected_frontier=_frontier(engine),
                              consumed_store_ids=list(ids), publication_excluded_store_ids=[],
                              filter_exclusion_proofs={}, carried_ranges=[])


def _ballast(tmp_path, *, covered=True, cover_tail=False, split_tool=False, uncovered_tool=False, raw_turns=3):
    """Store old turns summarized through F, an orphan tool row at F+1 (unbound, never shown), a uid-bound block,
    then raw turns. A node covers the block (without the tool row of turn 2 when ``split_tool``/``uncovered_tool``;
    with the fresh tail when ``cover_tail``); the frontier stays at F. Returns the facts the cases assert on."""
    clock = itertools.count(1_700_000_000)
    old, block, raw = _rows("old", 2, clock), _rows("cov", 14, clock), _rows("raw", raw_turns, clock)
    engine = _engine(tmp_path)
    try:
        _publish(engine, old, _store(engine, old), OLD_SUMMARY)
        frontier = _frontier(engine)
        orphan = engine._store.append("S", {"role": "tool", "content": "orphan tool output",
                                            "tool_call_id": "orphan_call"}, conversation_id="conv")
        assert orphan == frontier + 1
        block_ids, raw_ids = _store(engine, block), _store(engine, raw)
        node_rows = list(zip(block, block_ids))
        if split_tool or uncovered_tool:
            node_rows.pop(4 * 2 + 2)  # the tool result of turn 2: its assistant tool call is covered, it is not
        if cover_tail:
            node_rows += list(zip(raw, raw_ids))[-4:]
        if covered:
            _publish(engine, [r for r, _ in node_rows], [s for _, s in node_rows], COVERED_SUMMARY)
        assert _frontier(engine) == frontier  # the orphan keeps F where it is
    finally:
        engine.shutdown()
    view = [{"role": "system", "content": "sys"}, *[dict(row) for row in block + raw]]
    if split_tool:  # a result the store never saw under a never-bound uid: an unresolved row in the covered block
        view[1 + 4 * 2 + 2].update(message_uid=UNBOUND_UID, content="a tool result LCM never stored")
    return view, frontier, block_ids, raw_ids, {r["message_uid"] for r, _ in node_rows}


def _store_digest(engine, through=None):
    """(rows, content bytes, last store id) of the session, over the rows stored up to ``through`` when given."""
    return engine._store._conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(LENGTH(CAST(content AS BLOB))), 0), COALESCE(MAX(store_id), 0) "
        "FROM messages WHERE session_id = ? AND store_id <= ?", ("S", through or 1 << 62)).fetchone()


def _nodes(engine):
    return {n.node_id: (n.depth, n.summary, tuple(n.source_ids), n.source_type) for n in engine._dag.get_session_nodes("S")}


def _represented(nodes, node_id, text):
    """The node's summary is in ``text``, or a node that (transitively) condenses it is."""
    seen = {node_id}
    while seen:
        if any(nodes[n][1] in text for n in seen):
            return True
        seen = {p for p, (_d, _s, sources, kind) in nodes.items() if kind == "nodes" and set(sources) & seen}
    return False


def _compress(tmp_path, view, **kwargs):
    engine = _engine(tmp_path)  # cold resume: the host replays its whole view
    try:
        before, nodes_before = _store_digest(engine), _nodes(engine)
        result = engine.compress(view, **kwargs)
        facts = {
            "result": result, "before": before, "after": _store_digest(engine, before[2]), "nodes_before": nodes_before,
            "sweep": dict(engine._last_threshold_full_sweep or {}),
            "nodes_after": _nodes(engine), "frontier": _frontier(engine), "status": engine._uid_coverage_status(),
            "reason": engine._last_compression_noop_reason, "hold": engine._no_progress_hold,
            "hold_record": engine._store.read_metadata_json("compaction_no_progress:v1:S"),
        }
        facts["second"] = engine.compress(result)
        facts["nodes_second"] = _nodes(engine)
    finally:
        engine.shutdown()
    facts["new_sources"] = {sid for nid, (depth, _s, sources, kind) in facts["nodes_after"].items()
                            if nid not in facts["nodes_before"] and kind == "messages" for sid in sources}
    return facts


def _uids(rows):
    return [m.get("message_uid") for m in rows if m.get("message_uid")]


def _assert_lossless(facts):
    assert facts["after"] == facts["before"]  # stored rows: same count, same content bytes (ingest may add new ones)
    assert all(facts["nodes_after"].get(nid) == node for nid, node in facts["nodes_before"].items())  # no node rewritten


def _assert_tool_groups_whole(rows):
    calls = {c["id"] for m in rows if m.get("role") == "assistant" for c in m.get("tool_calls") or ()}
    results = {m.get("tool_call_id") for m in rows if m.get("role") == "tool"}
    assert calls == results, (calls ^ results)


def test_covered_block_is_dropped_and_uncovered_rows_kept(tmp_path):
    view, frontier, block_ids, raw_ids, covered = _ballast(tmp_path)
    facts = _compress(tmp_path, view)
    result, shown = facts["result"], set(_uids(facts["result"]))
    assert not shown & covered, sorted(shown & covered)[:5]  # every covered group left the host view
    assert not facts["new_sources"] & set(block_ids)  # and no leaf summarized it again
    raw_uids = [m["message_uid"] for m in view if m.get("message_uid", "").startswith("raw-")]
    assert all(uid in shown or sid in facts["new_sources"] for uid, sid in zip(raw_uids, raw_ids))
    assert _uids(result)[-4:] == raw_uids[-4:]  # the fresh tail, as the host sent it
    text = json.dumps(result)
    covering = next(nid for nid, node in facts["nodes_before"].items() if node[1] == COVERED_SUMMARY)
    assert _represented(facts["nodes_after"], covering, text)  # the covered rows live on through the DAG
    covered_tokens = count_messages_tokens([m for m in view if m.get("message_uid") in covered])
    assert count_messages_tokens(result) <= count_messages_tokens(view) - covered_tokens + 1_000, (
        count_messages_tokens(result), count_messages_tokens(view), covered_tokens)
    _assert_tool_groups_whole(result)
    _assert_lossless(facts)
    # Accounting: the orphan still holds F; the frontier never passes an unaccounted row.
    sources = {sid for (_d, _s, ids, kind) in facts["nodes_after"].values() if kind == "messages" for sid in ids}
    assert facts["frontier"] == frontier or all(sid in sources for sid in range(frontier + 1, facts["frontier"] + 1))
    assert facts["hold"] is None and facts["hold_record"] is None  # a pass that dropped rows is progress


def test_partly_covered_tool_group_is_kept_whole(tmp_path, monkeypatch):
    view, _frontier_id, _block_ids, _raw_ids, covered = _ballast(tmp_path, split_tool=True)
    facts = _compress(tmp_path, view)
    result, shown = facts["result"], set(_uids(facts["result"]))
    call_uid = view[1 + 4 * 2 + 1]["message_uid"]  # the covered assistant tool call of turn 2
    assert {call_uid, UNBOUND_UID} <= shown  # kept whole: its result the store never covered keeps it live
    assert not (shown & covered) - {call_uid}  # every other covered group is gone
    today = tmp_path / "today"
    today.mkdir()
    view, *_ = _ballast(today, split_tool=True)
    monkeypatch.setattr(lcm_engine.LCMEngine, "_drop_covered_host_groups", lambda self, state, leaf: None)
    baseline = _compress(today, view)
    assert facts["status"].get("unpaired_count") == baseline["status"].get("unpaired_count")  # as today
    _assert_tool_groups_whole(result)
    _assert_lossless(facts)


def test_group_with_uncovered_mapped_row_is_never_dropped(tmp_path):
    view, _frontier_id, block_ids, _raw_ids, covered = _ballast(tmp_path, uncovered_tool=True)
    facts = _compress(tmp_path, view)
    result, shown = facts["result"], set(_uids(facts["result"]))
    call_uid, tool_uid = view[1 + 4 * 2 + 1]["message_uid"], view[1 + 4 * 2 + 2]["message_uid"]
    tool_id = block_ids[4 * 2 + 2]
    # The group is either kept whole in the view, or its uncovered row went into a new leaf: never dropped unsummarized.
    assert {call_uid, tool_uid} <= shown or tool_id in facts["new_sources"], (sorted(shown)[:6], facts["new_sources"])
    assert not (shown & covered) - {call_uid}
    _assert_tool_groups_whole(result)
    _assert_lossless(facts)


def test_covered_fresh_tail_is_never_dropped(tmp_path):
    view, _frontier_id, _block_ids, _raw_ids, covered = _ballast(tmp_path, cover_tail=True)
    facts = _compress(tmp_path, view)
    shown = _uids(facts["result"])
    tail = _uids(view)[-4:]
    assert shown[-4:] == tail  # covered, but the fresh tail
    assert not set(shown) & (covered - set(tail))
    _assert_lossless(facts)


def test_view_without_uids_keeps_todays_rows(tmp_path):
    view, *_ = _ballast(tmp_path)
    plain = [{k: v for k, v in m.items() if k != "message_uid"} for m in view]
    facts = _compress(tmp_path, plain)
    kept = [m for m in facts["result"] if str(m.get("content", "")).startswith("cov ")]
    block = [m for m in plain if str(m.get("content", "")).startswith("cov ")]
    assert kept == block[len(block) - len(kept):] and kept, (len(kept), len(block))  # no covered row dropped
    assert facts["after"] == facts["before"]
    assert all(facts["nodes_after"].get(nid) == node for nid, node in facts["nodes_before"].items())


def test_second_pass_is_a_noop_and_nothing_covered_reads_nothing_more(tmp_path, monkeypatch):
    view, *_ = _ballast(tmp_path)
    facts = _compress(tmp_path, view)
    second = facts["second"]
    assert isinstance(second, list) and set(_uids(second)) <= set(_uids(facts["result"]))
    assert all(facts["nodes_second"].get(nid) == node for nid, node in facts["nodes_after"].items())

    calls = []
    original = lcm_engine.LCMEngine._store_complete_node_covered

    def counting(self, store_ids, *args, **kwargs):
        calls.append((args, kwargs))
        return original(self, store_ids, *args, **kwargs)

    monkeypatch.setattr(lcm_engine.LCMEngine, "_store_complete_node_covered", counting)
    bare = tmp_path / "bare"
    bare.mkdir()
    view, *_ = _ballast(bare, covered=False)
    _compress(bare, view)
    assert not [call for call in calls if call[1].get("session_id")]  # nothing covered: no extra coverage read

    calls.clear()
    swept = tmp_path / "swept"
    swept.mkdir()
    view, *_ = _ballast(swept, raw_turns=8)  # several leaf passes in one sweep
    swept_facts = _compress(swept, view)
    assert swept_facts["sweep"].get("leaf_passes", 0) > 1, swept_facts["sweep"]
    assert len([call for call in calls if call[1].get("session_id")]) == 1  # once, at the first leaf pass


def test_cold_gap_request_reaches_the_drop(tmp_path):
    view, _frontier_id, _block_ids, _raw_ids, covered = _ballast(tmp_path)
    engine = _engine(tmp_path)
    try:
        engine.threshold_tokens = count_messages_tokens(view) * 2  # below the threshold: only the cold request runs
        engine._compaction_request.cold_gap = True
        result = engine.compress(view, current_tokens=count_messages_tokens(view))
        cold = engine._cold_gap_status()
    finally:
        engine.shutdown()
    assert not set(_uids(result)) & covered
    assert len(result) < len(view), cold
