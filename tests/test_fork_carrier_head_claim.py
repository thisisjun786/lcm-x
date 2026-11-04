"""fork: an LCM summary carrier at the head of the host view must claim the rows it absorbed (synthetic fixture).

A long Telegram session's host view began with ONE user row the host built itself (survival-fit / boundary rewrite):
verified LCM summary parts of this session in a combination LCM never emitted, ``\\n\\n``, then a short one-line user
remainder. The row's own ``message_uid`` was an engine uid (``kind='engine'``, ``store_id=0``);
``_absorbed_message_uids`` named one host uid, canonically bound to the stored user row R above the frontier whose
bytes are the remainder; the carrier's host stamp was a day later than R's. Under the v4 emission proof the carrier
keeps its full identity, so no view row maps R. Every other view row was uid-bound above the frontier. The carrier
alone was the oldest leaf chunk (larger than the chunk budget) and claimed nothing (no ``\\n\\n`` inside the
remainder, no row at its stamp), so the uid leaf builder dropped it and every pass ended ``no paired uncovered
source rows`` with zero leaves. The carrier now claims the rows its absorbed uids name when their stored text is
exactly its remainder.
"""

from __future__ import annotations

import itertools

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens
from tests.test_fork_covered_head import _rows, _store
from tests.test_fork_late_copy_mapping import _home
from tests.test_fork_rewritten_uid_mapping import bind

SUMMARY_PAD = " the operator reviewed the deployment plan and its rollback notes" * 240
REMAINDER = "ok, go ahead."
CARRIER_UID = "lcm-engine-carrier-0"
CHUNK = 1500  # below the carrier's size under either tokenizer (char estimate or tiktoken)


def _engine(tmp_path, *, fresh_tail_count=4, chunk=CHUNK) -> LCMEngine:
    engine = LCMEngine(config=LCMConfig(fresh_tail_count=fresh_tail_count, leaf_chunk_tokens=chunk,
                                        dynamic_leaf_chunk_max=chunk, database_path=str(tmp_path / "lcm.db")),
                       hermes_home=_home(tmp_path))
    engine.on_session_start("S", platform="telegram", context_length=12_000, conversation_id="conv")
    return engine


def _part(node) -> str:
    label = {0: "Recent", 1: "Session Arc", 2: "Durable"}[node.depth]
    return f"[{label} Summary (d{node.depth}, node {node.node_id})]\n{node.summary}\n[Expand for details: {node.expand_hint}]"


def _carrier_view(tmp_path, monkeypatch, remainder=REMAINDER):
    """Two real compactions (a v4 emission proof in force), then R and raw rows stored above the frontier.
    Returns (view, carrier, R's store id, raw store ids)."""
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
        raw = _rows("raw", 12, clock)
        raw_ids = _store(engine, raw)
        bind(engine, CARRIER_UID, 0, "engine")
        later = max((n for n in engine._dag.get_session_nodes("S") if n.depth == 0), key=lambda n: n.node_id)
    finally:
        engine.shutdown()
    assert frontier < r_id < raw_ids[0]
    carrier = {"role": "user", "content": _part(later) + "\n\n" + remainder,
               "timestamp": user["timestamp"] + 86_400.0, "message_uid": CARRIER_UID,
               "_absorbed_message_uids": ["host-uid-R"]}
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **kw: ("Synthetic leaf summary.", 1))
    return [{"role": "system", "content": "sys"}, carrier, *[dict(row) for row in raw]], carrier, r_id, raw_ids


def _compress(tmp_path, view):
    engine = _engine(tmp_path)  # cold resume: the host replays its whole view
    try:
        stored_before = engine._store.get_session_messages("S")
        nodes_before = {n.node_id for n in engine._dag.get_session_nodes("S")}
        tokens = count_messages_tokens(view)
        result = engine.compress(view)
        out = {
            "result": result, "tokens": tokens, "reason": engine._last_compression_noop_reason,
            "new_leaves": [n for n in engine._dag.get_session_nodes("S") if n.depth == 0 and n.node_id not in nodes_before],
            "status": engine._uid_coverage_status(), "remainder": engine._generated_context_carrier_remainder(view[1]),
            "stored_before": stored_before, "stored_after": engine._store.get_session_messages("S"),
        }
    finally:
        engine.shutdown()
    return out


def test_carrier_head_claims_its_absorbed_stored_row(tmp_path, monkeypatch):
    view, carrier, r_id, raw_ids = _carrier_view(tmp_path, monkeypatch)
    assert count_messages_tokens([carrier]) > CHUNK  # the carrier alone is the oldest leaf chunk
    run = _compress(tmp_path, view)
    assert run["remainder"] == REMAINDER
    assert run["reason"] != "no paired uncovered source rows", run["reason"]
    assert run["new_leaves"], run["reason"]
    sources = [sid for node in run["new_leaves"] for sid in node.source_ids]
    assert sources.count(r_id) == 1, sources  # R is published once, through the carrier's claim
    assert set(sources) <= {r_id, *raw_ids}
    result = run["result"]
    assert len(result) < len(view) and count_messages_tokens(result) < run["tokens"], (
        len(result), len(view), count_messages_tokens(result), run["tokens"])
    assert run["status"].get("unpaired_count", 0) == 0, run["status"]
    after = {int(row["store_id"]): row for row in run["stored_after"]}
    assert all(after.get(int(row["store_id"])) == row for row in run["stored_before"])  # lossless


def test_carrier_with_other_remainder_claims_nothing(tmp_path, monkeypatch):
    """The absorbed uid names R but the remainder is other text: no claim. As before, ingest stores the carrier
    as a new row of its own; R is never published by the carrier's leaf, nor twice."""
    view, _carrier, r_id, raw_ids = _carrier_view(tmp_path, monkeypatch, remainder="ok, do not go ahead.")
    run = _compress(tmp_path, view)
    assert run["remainder"] == "ok, do not go ahead."
    stored = {int(row["store_id"]) for row in run["stored_before"]}
    new_rows = [row for row in run["stored_after"] if int(row["store_id"]) not in stored]
    carrier_ids = [int(row["store_id"]) for row in new_rows if row["content"].endswith("\n\nok, do not go ahead.")]
    assert len(carrier_ids) == 1, new_rows
    leaves = run["new_leaves"]
    assert [sid for node in leaves for sid in node.source_ids].count(r_id) <= 1
    assert not any(r_id in node.source_ids and carrier_ids[0] in node.source_ids for node in leaves)
