"""Fork cold-cache compaction (lina#455): every cause of a cold provider call asks for the cold pass, the one-shot
event survives an interrupted pass, a pass that frees nothing is retried once through the threshold path and is
never a silent full-size call, and a leaf valve skips oldest chunks nothing can claim.

Synthetic fixtures only; time comes from a fake ``time.time`` (never a sleep). Token numbers are compared under the
character estimate (``character_estimate`` stubs ``tokens._get_encoder`` and ``_encoder_generation`` together).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm import compaction_request
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.escalation import SweepBudgetExhausted
from hermes_lcm.tokens import count_messages_tokens
from hermes_lcm import tokens as lcm_tokens
from tests.test_cold_gap_compaction import CONVERSATION, GAP, T0, THRESHOLD, _Clock, _messages, _provider_call, _status
from tests.test_fork_carrier_head_claim import CARRIER_UID, CHUNK
from tests.test_fork_carrier_head_claim import _engine as _valve_engine
from tests.test_fork_spent_carrier_head import _spent_carrier_view
from tests.test_issue_671_stub_first_exit import payload_of, tool_pair

REQUESTED = "LCM cold-gap compaction requested:"
FREED_NOTHING = "LCM cold-cache compaction freed nothing:"
VALVE = "LCM leaf valve skipped an unusable oldest chunk:"
SYSTEM_A = {"role": "system", "content": "You are the synthetic assistant. Prompt A."}
SYSTEM_B = {"role": "system", "content": [{"type": "text", "text": "You are the synthetic assistant. Prompt B."}]}


@pytest.fixture(autouse=True)
def character_estimate(monkeypatch):
    monkeypatch.setattr(lcm_tokens, "_get_encoder", lambda: None)
    monkeypatch.setattr(lcm_tokens, "_encoder_generation", -1)


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock(T0)
    monkeypatch.setattr(time, "time", fake)
    return fake


def _engine(tmp_path, monkeypatch, *, summary="bounded", **overrides) -> LCMEngine:
    settings = dict(fresh_tail_count=2, leaf_chunk_tokens=100, threshold_full_sweep_enabled=True,
                    cold_gap_compaction_seconds=GAP, database_path=str(tmp_path / "lcm.db"))
    settings.update(overrides)
    engine = LCMEngine(config=LCMConfig(**settings), hermes_home=str(tmp_path / "hermes_home"))
    engine.on_session_start(CONVERSATION)
    engine.update_model("model-a", 1_000_000, provider="provider-a", api_mode="responses")
    engine.threshold_tokens = THRESHOLD

    def fake_leaf(chunk, focus_topic=None, deadline=None):
        del focus_topic, deadline
        return chunk, count_messages_tokens(chunk), summary, 1, 0

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", fake_leaf)
    return engine


@pytest.fixture
def engines(tmp_path, monkeypatch):
    built = []

    def build(name="e", **overrides):
        engine = _engine(tmp_path / name, monkeypatch, **overrides)
        built.append(engine)
        return engine

    yield build
    for engine in built:
        engine.shutdown()


def _record(engine) -> dict:
    return engine._store.read_metadata_json(engine._cold_gap_metadata_key(CONVERSATION)) or {}


def _event(engine) -> dict:
    return _record(engine).get("last_event") or {}


def _requested_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(REQUESTED)]


def _switch_model(engine, model="model-b", provider="provider-a") -> None:
    engine.update_model(model, 1_000_000, provider=provider, api_mode="responses")
    engine.threshold_tokens = THRESHOLD


def _with_prompt(system, messages):
    return [system, *messages[1:]]


# -- 2. model_change -------------------------------------------------------------------------------------------------


def test_model_change_requests_the_cold_pass(engines, clock, caplog):
    engine = engines()
    _provider_call(engine, clock, T0, prompt_tokens=400_000)
    clock.now = T0 + 5
    _switch_model(engine)
    messages = _messages()
    with caplog.at_level(logging.INFO):
        assert engine.should_compress_preflight(messages) is True
    assert engine._compaction_request.cold_gap is True
    assert [line.endswith("cause=model_change requests=1") for line in _requested_lines(caplog)] == [True]
    assert _event(engine)["cause"] == "model_change" and _event(engine)["outcome"] == "requested"
    result = engine.compress(messages, current_tokens=count_messages_tokens(messages))
    assert len(result) < len(messages)
    assert _event(engine)["outcome"] == "shrunk" and _event(engine)["attempts"] == 1


@pytest.mark.parametrize("model, provider", [("model-b", "provider-a"), ("model-a", "provider-b")])
def test_model_or_provider_change_alone_triggers(engines, clock, model, provider):
    engine = engines()
    _provider_call(engine, clock, T0)
    clock.now = T0 + 5
    _switch_model(engine, model, provider)
    assert engine.should_compress_preflight(_messages()) is True
    assert _event(engine)["cause"] == "model_change"


def test_same_model_does_not_trigger(engines, clock):
    engine = engines()
    _provider_call(engine, clock, T0)
    clock.now = T0 + 5
    _switch_model(engine, "model-a")
    assert engine.should_compress_preflight(_messages()) is False
    assert engine._compaction_request.cold_gap is False and _record(engine) == {}


def test_model_change_below_the_floor_does_not_trigger(engines, clock):
    engine = engines()
    _provider_call(engine, clock, T0, prompt_tokens=299_999)
    clock.now = T0 + 5
    _switch_model(engine)
    assert engine.should_compress_preflight(_messages()) is False
    assert _record(engine) == {}


# -- 3. prompt_change through select_context ------------------------------------------------------------------------


def _prompt_baseline(engine, clock, system=SYSTEM_A):
    """A provider call sent with ``system``: select_context observed it, the response hook recorded it."""
    assert engine.select_context(_with_prompt(system, _messages())) is None
    _provider_call(engine, clock, T0)
    clock.now = T0 + 5


def test_prompt_change_requests_through_the_pre_api_gate(engines, clock, caplog):
    engine = engines()
    _prompt_baseline(engine, clock)
    recorded = engine._store.read_compaction_telemetry(CONVERSATION)["last_system_prompt_sha"]
    assert recorded == compaction_request.system_prompt_sha([SYSTEM_A])
    request = _with_prompt(SYSTEM_B, _messages())
    assert engine.select_context(request, conversation_messages=request[1:], budget_tokens=1) is None
    with caplog.at_level(logging.INFO):
        assert engine.should_compress(count_messages_tokens(request)) is True
    assert [line.endswith("cause=prompt_change requests=1") for line in _requested_lines(caplog)] == [True]
    result = engine.compress(request, current_tokens=count_messages_tokens(request))
    assert len(result) < len(request)
    assert _event(engine)["cause"] == "prompt_change" and _event(engine)["outcome"] == "shrunk"
    # The host's loop re-enters the gate after the pass: the event is consumed.
    assert engine.select_context(result) is None
    assert engine.should_compress(count_messages_tokens(result)) is False


def test_identical_prompt_does_not_trigger(engines, clock):
    engine = engines()
    _prompt_baseline(engine, clock)
    request = _with_prompt(dict(SYSTEM_A), _messages())
    assert engine.select_context(request) is None
    assert engine.should_compress(count_messages_tokens(request)) is False
    assert _record(engine) == {}


def test_prompt_hash_is_recorded_only_after_a_provider_response(engines, clock):
    engine = engines()
    _prompt_baseline(engine, clock)
    engine.select_context(_with_prompt(SYSTEM_B, _messages()))
    telemetry = engine._store.read_compaction_telemetry(CONVERSATION)
    assert telemetry["last_system_prompt_sha"] == compaction_request.system_prompt_sha([SYSTEM_A])
    _provider_call(engine, clock, T0 + 10)
    telemetry = engine._store.read_compaction_telemetry(CONVERSATION)
    assert telemetry["last_system_prompt_sha"] == compaction_request.system_prompt_sha([SYSTEM_B])
    assert telemetry["last_api_call_at"] == T0 + 10


def test_select_context_returns_none_and_never_mutates(engines, clock):
    engine = engines()
    request = _with_prompt(SYSTEM_B, _messages())
    conversation = copy.deepcopy(request[1:])
    before, conversation_before = copy.deepcopy(request), copy.deepcopy(conversation)
    assert engine.select_context(request, conversation_messages=conversation, incoming_message=request[-1],
                                 budget_tokens=10) is None
    assert request == before and conversation == conversation_before


def test_select_context_never_raises(engines, monkeypatch):
    engine = engines()

    def broken(messages):
        raise RuntimeError("synthetic hash failure")

    monkeypatch.setattr(lcm_engine, "system_prompt_sha", broken)
    assert engine.select_context(_with_prompt(SYSTEM_A, _messages())) is None
    assert engine.select_context(object()) is None  # not even a list


@pytest.mark.parametrize("kind", ["stateless", "ignored"])
def test_select_context_is_a_no_op_for_a_bypassed_session(engines, clock, kind):
    engine = engines()
    _prompt_baseline(engine, clock)
    setattr(engine, f"_session_{kind}", True)
    assert engine.select_context(_with_prompt(SYSTEM_B, _messages())) is None
    assert engine._pending_system_prompt_sha_for_conversation() == compaction_request.system_prompt_sha([SYSTEM_A])
    assert engine.should_compress(1000) is False


def test_system_prompt_sha_reads_the_leading_system_rows_only():
    blocks = [{"role": "system", "content": [{"type": "text", "text": "a"}]}, {"role": "system", "content": "b"}]
    sha = compaction_request.system_prompt_sha(blocks + [{"role": "user", "content": "x"}])
    assert sha == compaction_request.system_prompt_sha(blocks + [{"role": "user", "content": "y"}])
    assert sha != compaction_request.system_prompt_sha(blocks[:1] + [{"role": "user", "content": "x"}])
    assert compaction_request.system_prompt_sha([{"role": "user", "content": "x"}]) is None


# -- 4. suppression, no telemetry, threshold ------------------------------------------------------------------------


@pytest.mark.parametrize("cause", ["model", "prompt"])
def test_a_compaction_committed_after_the_last_call_suppresses_model_and_prompt(engines, clock, cause):
    engine = engines()
    _prompt_baseline(engine, clock)
    messages = _messages()
    engine.compress(messages, current_tokens=count_messages_tokens(messages), force=True)  # commits leaves
    assert engine.compression_count >= 1
    if cause == "model":
        _switch_model(engine)
        assert engine.should_compress_preflight(messages) is False
    else:
        engine.select_context(_with_prompt(SYSTEM_B, messages))
        assert engine.should_compress(count_messages_tokens(messages)) is False
    assert _record(engine) == {}


def test_no_telemetry_never_triggers(engines, clock):
    engine = engines()
    engine.last_prompt_tokens = 400_000
    _switch_model(engine)
    engine.select_context(_with_prompt(SYSTEM_B, _messages()))
    assert engine._store.read_compaction_telemetry(CONVERSATION) is None
    assert engine.should_compress_preflight(_messages()) is False
    assert engine.should_compress(1000) is False


def test_at_or_over_the_threshold_the_ordinary_path_runs(engines, clock):
    engine = engines()
    _provider_call(engine, clock, T0)
    clock.now = T0 + 5
    _switch_model(engine)
    messages = _messages()
    engine.threshold_tokens = count_messages_tokens(messages) - 1
    assert engine.should_compress_preflight(messages) is True
    assert engine._compaction_request.cold_gap is False and _record(engine) == {}


# -- 5. one-shot bookkeeping -----------------------------------------------------------------------------------------


def test_a_finished_pass_does_not_refire_for_the_same_call(engines, clock):
    engine = engines()
    _provider_call(engine, clock, T0)
    clock.now = T0 + GAP
    messages = _messages()
    assert engine.should_compress_preflight(messages) is True
    engine.compress(messages, current_tokens=count_messages_tokens(messages))
    assert _event(engine)["outcome"] == "shrunk"
    clock.now = T0 + 3 * GAP
    _switch_model(engine)
    assert engine.should_compress_preflight(messages) is False
    assert engine.should_compress(count_messages_tokens(messages)) is False


def test_an_interrupted_pass_asks_again_at_most_twice(engines, clock, monkeypatch):
    engine = engines()
    _provider_call(engine, clock, T0)
    clock.now = T0 + GAP
    messages = _messages()

    def interrupted(*args, **kwargs):
        raise RuntimeError("synthetic interruption")

    monkeypatch.setattr(engine, "_compress_impl", interrupted)
    for request in (1, 2):
        assert engine.should_compress_preflight(messages) is True
        with pytest.raises(RuntimeError):
            engine.compress(messages, current_tokens=count_messages_tokens(messages))
        event = _event(engine)
        assert (event["requests"], event["outcome"], event["pass_started"]) == (request, "error", True)
    assert engine.should_compress_preflight(messages) is False
    assert engine.should_compress(count_messages_tokens(messages)) is False


HOST_ITERATIONS = 6


@pytest.mark.parametrize("outcome, gate_true", [("shrunk", 1), ("no_progress", 1), ("error", 2)])
def test_the_pre_api_gate_never_loops_within_one_turn(engines, clock, monkeypatch, outcome, gate_true):
    # The host calls should_compress(tokens) once per provider request iteration and compresses when it says True;
    # no provider call lands in between, so every iteration sees the same last call.
    engine = engines()
    messages = _messages(rows=2) if outcome == "no_progress" else _messages()
    tokens = count_messages_tokens(messages)
    if outcome == "no_progress":
        engine.threshold_tokens = int(tokens / 0.97)
    if outcome == "error":
        def interrupted(*args, **kwargs):
            raise RuntimeError("synthetic interruption")

        monkeypatch.setattr(engine, "_compress_impl", interrupted)
    _provider_call(engine, clock, T0)
    clock.now = T0 + GAP
    gate = []
    for _iteration in range(HOST_ITERATIONS):
        tokens = count_messages_tokens(messages)
        gate.append(engine.should_compress(tokens))
        if gate[-1]:
            try:
                messages = engine.compress(messages, current_tokens=tokens)
            except RuntimeError:
                pass
    assert gate == [True] * gate_true + [False] * (HOST_ITERATIONS - gate_true), gate
    event = _event(engine)
    assert (event["outcome"], event["requests"]) == (outcome, gate_true)


def test_a_requested_record_left_by_a_dead_pass_asks_again(engines, clock):
    engine = engines()
    _provider_call(engine, clock, T0)
    clock.now = T0 + GAP
    last_call_at = engine._store.read_compaction_telemetry(CONVERSATION)["last_api_call_at"]
    dead = {"requested_at": T0 + GAP - 30, "cause": "idle", "requests": 1, "attempts": 1, "outcome": "requested",
            "pass_started": True, "owner": "999999:deadbeef0000", "stop_reason": "requested"}
    engine._store.write_metadata_json([engine._cold_gap_metadata_key(CONVERSATION)], json.dumps(
        {"fired_for_last_api_call_at": last_call_at, "last_event": dead}, sort_keys=True))
    messages = _messages()
    assert engine.should_compress_preflight(messages) is True
    assert _event(engine)["requests"] == 2
    engine.compress(messages, current_tokens=count_messages_tokens(messages))
    assert _event(engine)["outcome"] == "shrunk"
    assert engine.should_compress_preflight(messages) is False


def test_a_request_this_process_is_still_running_is_not_asked_again(engines, clock):
    engine = engines()
    _provider_call(engine, clock, T0)
    clock.now = T0 + GAP
    messages = _messages()
    assert engine.should_compress_preflight(messages) is True
    engine._update_cold_gap_event(pass_started=True)  # the pass is in flight in this process
    assert engine.should_compress(count_messages_tokens(messages)) is False
    assert _event(engine)["requests"] == 1


# -- 6.-8. no-progress retry, both attempts empty, nothing truncated -------------------------------------------------


def _store_digest(engine) -> tuple[int, str]:
    rows = engine._store.get_session_messages(engine._session_id)
    body = json.dumps([[row["store_id"], row["role"], row.get("content")] for row in rows], sort_keys=True,
                      default=str)
    return len(rows), hashlib.sha256(body.encode()).hexdigest()


def _tool_conversation() -> list[dict]:
    """Large (aged-stub sized) tool outputs outside the fresh tail; ingest does not externalize them."""
    messages = [{"role": "system", "content": "system"}]
    for turn in range(6):
        messages.append({"role": "user", "content": f"ask {turn} " + "word " * 20})
        messages += tool_pair(f"call-{turn}", payload_of(2_600, word=f"w{turn}"))
        messages.append({"role": "assistant", "content": f"reply {turn} " + "word " * 20})
    return messages


def _out_of_time(engine, monkeypatch) -> list[int]:
    """Every leaf call stops on the foreground time budget: the cold pass stores no leaf."""
    calls: list[int] = []

    def out_of_time(chunk, focus_topic=None, deadline=None):
        calls.append(len(chunk))
        raise SweepBudgetExhausted("time_budget_exhausted")

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", out_of_time)
    return calls


def test_a_cold_pass_that_frees_nothing_is_retried_through_the_threshold_path(engines, clock, monkeypatch, caplog):
    engine = engines(leaf_chunk_tokens=1_000)
    messages = _tool_conversation()
    tokens = count_messages_tokens(messages)
    engine.threshold_tokens = int(tokens / 0.97)  # ~97% of the threshold
    calls = _out_of_time(engine, monkeypatch)
    _provider_call(engine, clock, T0)
    clock.now = T0 + GAP
    assert engine.should_compress_preflight(messages) is True
    stored_before = _store_digest(engine)
    with caplog.at_level(logging.INFO):
        result = engine.compress(messages, current_tokens=tokens)
    assert calls  # the cold attempt tried a leaf and ran out of time
    assert len(result) <= len(messages) and count_messages_tokens(result) < 0.95 * tokens
    event = _event(engine)
    assert (event["cause"], event["attempts"], event["outcome"]) == ("idle", 2, "shrunk")
    assert event["tokens_before"] == tokens and event["tokens_after"] == count_messages_tokens(result)
    assert not any(r.getMessage().startswith(FREED_NOTHING) for r in caplog.records)
    # 8: no hold, nothing dropped or truncated: the store is byte-identical and every host row is still there
    # (large tool outputs are replaced by recoverable stubs only).
    assert engine._no_progress_hold is None and not engine._sweep_budget_hold_active()
    assert _store_digest(engine) == stored_before
    assert [m["role"] for m in result] == [m["role"] for m in messages]
    assert result[0]["content"].startswith(messages[0]["content"])  # the system row only gains LCM's note
    stubbed = [m for m, original in zip(result[1:], messages[1:]) if m.get("content") != original.get("content")]
    assert stubbed and all(m["role"] == "tool" and m["content"].startswith("[Externalized tool output:")
                           for m in stubbed)
    assert not (_status(engine).get("last_survival_fit") or {}).get("dropped_rows")


def test_when_both_attempts_free_nothing_the_original_list_returns_and_is_reported(engines, clock, caplog):
    engine = engines()
    messages = _messages(rows=2)  # every row is in the fresh tail: nothing to summarise, nothing to stub
    tokens = count_messages_tokens(messages)
    engine.threshold_tokens = int(tokens / 0.97)
    _provider_call(engine, clock, T0)
    clock.now = T0 + GAP
    assert engine.should_compress_preflight(messages) is True
    stored_before = _store_digest(engine)
    with caplog.at_level(logging.INFO):
        result = engine.compress(messages, current_tokens=tokens)
    assert result == messages
    warnings = [r for r in caplog.records if r.getMessage().startswith(FREED_NOTHING)]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING
    assert "cause=idle attempts=2 stop=" in warnings[0].getMessage()
    event = _event(engine)
    assert (event["attempts"], event["outcome"]) == (2, "no_progress")
    assert event["tokens_before"] == event["tokens_after"] == tokens
    assert engine._no_progress_hold is None and engine._no_progress_hold_status() is None
    assert not engine._sweep_budget_hold_active()
    assert not (engine._zero_leaf_streak_status() or {}).get("passes")
    assert _store_digest(engine) == stored_before
    assert not (_status(engine).get("last_survival_fit") or {}).get("dropped_rows")
    # Consumed: the same call never asks again; the next ordinary threshold pass still runs.
    assert engine.should_compress_preflight(messages) is False
    grown = messages + _messages()[1:]
    engine.threshold_tokens = count_messages_tokens(grown) - 1
    assert engine.should_compress_preflight(grown) is True
    assert engine._compaction_request.cold_gap is False
    shrunk = engine.compress(grown, current_tokens=count_messages_tokens(grown))
    assert len(shrunk) < len(grown)


def test_a_shrinking_cold_pass_covers_every_row_it_removes(engines, clock):
    engine = engines()
    messages = _messages()
    _provider_call(engine, clock, T0)
    clock.now = T0 + GAP
    assert engine.should_compress_preflight(messages) is True
    stored_before = _store_digest(engine)
    result = engine.compress(messages, current_tokens=count_messages_tokens(messages))
    assert _store_digest(engine) == stored_before
    kept = {m.get("content") for m in result}
    rows = {row["content"]: row["store_id"] for row in engine._store.get_session_messages(engine._session_id)}
    covered = {sid for node in engine._dag.get_session_nodes(engine._session_id) for sid in node.source_ids}
    removed = [m for m in messages[1:] if m["content"] not in kept]
    assert removed and all(rows[m["content"]] in covered for m in removed)
    assert not (_status(engine).get("last_survival_fit") or {}).get("dropped_rows")


# -- 9. structural leaf valve ----------------------------------------------------------------------------------------


def _without_scaffold_exclusion(engine, monkeypatch):
    """Disable the d8a937e3 first line: unmapped LCM scaffold stays among the leaf candidates again."""
    original = engine._uid_leaf_candidates

    def candidates(candidate, view, full_map):
        kept = {id(m) for m in original(candidate, view, full_map)}
        return [m for m in candidate if id(m) in kept or engine._generated_context_carrier_remainder(m) is not None
                or engine._is_context_summary_content(m.get("content"))
                or engine._preserved_objective_context_content(m)]

    monkeypatch.setattr(engine, "_uid_leaf_candidates", candidates)


def _head(shape, carrier):
    pad = " the operator reviewed the deployment plan and its rollback notes" * 240
    if shape == "carrier":
        return carrier
    if shape == "summary":
        return {"role": "user", "content": carrier["content"].rsplit("\n\n", 1)[0],  # the summary part alone
                "timestamp": carrier["timestamp"], "message_uid": CARRIER_UID}
    if shape == "objective":
        return {"role": "user", "content": "[Current user objective preserved from compacted history]\n" + pad,
                "timestamp": carrier["timestamp"], "message_uid": CARRIER_UID}
    # NEW shape: an engine-bound host uid (no stored row) on a row that is not LCM scaffold (d8a937e3 cannot see it)
    return {"role": "user", "content": "a host-built row LCM never stored" + pad, "timestamp": carrier["timestamp"],
            "message_uid": CARRIER_UID}


def _leaves_after(engine, view):
    nodes_before = {n.node_id for n in engine._dag.get_session_nodes("S")}
    result = engine.compress(view)
    new = [n for n in engine._dag.get_session_nodes("S") if n.depth == 0 and n.node_id not in nodes_before]
    return result, new


def _valve_case(path, monkeypatch, shape, *, valve, fresh_tail_count=4):
    """One fresh store per run: the shape's head row, then the uid-bound raw backlog of the spent-carrier view."""
    path.mkdir(parents=True)
    view, carrier, raw_ids, _r_id = _spent_carrier_view(path, monkeypatch)
    head = _head(shape, carrier)
    view = [view[0], head, *view[2:]]
    assert count_messages_tokens([head]) > CHUNK  # the head alone is the oldest leaf chunk
    engine = _valve_engine(path, fresh_tail_count=fresh_tail_count)
    try:
        _without_scaffold_exclusion(engine, monkeypatch)
        if not valve:
            monkeypatch.setattr(engine, "_leaf_valve_skip", lambda *args: False)
        stored_before = engine._store.get_session_messages("S")
        result, leaves = _leaves_after(engine, view)
        stored_after = {int(row["store_id"]): row for row in engine._store.get_session_messages("S")}
        reason = engine._last_compression_noop_reason
        sweep = dict(engine._last_threshold_full_sweep or {})
    finally:
        engine.shutdown()
    return view, head, raw_ids, result, leaves, (reason, sweep), (stored_before, stored_after)


def test_the_leaf_valve_skips_an_unusable_head_chunk(tmp_path, monkeypatch, caplog):
    # Without the valve (and without the d8a937e3 line) the carrier head stores no leaf: the fixture reaches it.
    _view, _head_row, _raw, _result, leaves, reason, _stored = _valve_case(
        tmp_path / "control", monkeypatch, "carrier", valve=False)
    assert not leaves and reason[0] == "no paired uncovered source rows", reason
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        view, head, raw_ids, result, leaves, reason, stored = _valve_case(
            tmp_path / "valve", monkeypatch, "carrier", valve=True)
    assert leaves, reason
    assert {sid for node in leaves for sid in node.source_ids} <= set(raw_ids)
    assert len(result) < len(view) and count_messages_tokens(result) < count_messages_tokens(view)
    assert all(stored[1].get(int(row["store_id"])) == row for row in stored[0])  # nothing deleted or changed
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith(VALVE)]
    assert len(lines) == 1 and "rows=1 store_ids=[null]" in lines[0]
    assert f"lengths=[{len(head['content'])}]" in lines[0] and head["content"][:40] not in lines[0]


@pytest.mark.parametrize("shape", ["summary", "objective", "engine_uid_row"])
def test_other_unclaimable_heads_never_stop_compaction(tmp_path, monkeypatch, caplog, shape):
    # A bare summary part, a preserved objective, and a non-scaffold row on an engine-bound uid at the head are
    # handled before chunk selection (leading scaffold, or ingest/replay of the row), with or without the d8a937e3
    # line and with or without the valve: the backlog behind them is summarised and the valve stays idle.
    for valve in (False, True):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            view, _head_row, _raw_ids, result, leaves, reason, stored = _valve_case(
                tmp_path / str(valve), monkeypatch, shape, valve=valve)
        assert leaves, reason
        assert len(result) < len(view)
        assert all(stored[1].get(int(row["store_id"])) == row for row in stored[0])
        assert not any(r.getMessage().startswith(VALVE) for r in caplog.records)


def test_a_view_with_nothing_usable_is_no_progress(tmp_path, monkeypatch, caplog):
    # The carrier head, then only fresh-tail rows: the valve skips the head and nothing is left to summarise.
    with caplog.at_level(logging.WARNING):
        view, _head_row, raw_ids, result, leaves, (reason, sweep), stored = _valve_case(
            tmp_path / "case", monkeypatch, "carrier", valve=True, fresh_tail_count=48)
    assert len(raw_ids) == 48  # every raw row is in the fresh tail
    assert not leaves  # (this 12k-window view is over the window: the ordinary survival fit still applies after)
    assert reason == "no paired uncovered source rows"  # the no-op it was, not a drained prefix
    assert sweep.get("stop_reason") in (None, "no_usable_leaf_input")
    assert [r.getMessage().startswith(VALVE) for r in caplog.records].count(True) == 1
    assert all(stored[1].get(int(row["store_id"])) == row for row in stored[0])


# -- 10. lcm_status and LCM_COLD_CACHE_CAUSES -------------------------------------------------------------------------


def test_lcm_status_reports_the_cold_cache_fields(engines, clock):
    engine = engines()
    _provider_call(engine, clock, T0)
    clock.now = T0 + 5
    _switch_model(engine)
    messages = _messages()
    assert engine.should_compress_preflight(messages) is True
    engine.compress(messages, current_tokens=count_messages_tokens(messages))
    block = _status(engine)["cold_gap_compaction"]
    assert block["causes"] == "idle,model,prompt"
    event = block["last_event"]
    assert (event["cause"], event["requests"], event["attempts"], event["outcome"]) == ("model_change", 1, 1, "shrunk")
    for key in ("tokens_before", "tokens_after", "duration_ms", "stop_reason", "requested_at", "gap_seconds"):
        assert key in event


@pytest.mark.parametrize("raw, expected, warned", [
    (None, "idle,model,prompt", False),
    ("prompt, idle", "idle,prompt", False),
    ("MODEL", "model", False),
    ("idle,bogus", "idle", True),
    ("bogus", "idle,model,prompt", True),
    ("", "idle,model,prompt", True),
    (" , ", "idle,model,prompt", True),
])
def test_cold_cache_causes_env(monkeypatch, raw, expected, warned):
    if raw is None:
        monkeypatch.delenv("LCM_COLD_CACHE_CAUSES", raising=False)
    else:
        monkeypatch.setenv("LCM_COLD_CACHE_CAUSES", raw)
    config = LCMConfig.from_env()
    assert config.cold_cache_causes == expected
    assert any("LCM_COLD_CACHE_CAUSES" in w for w in config.config_source_warnings) == warned


def test_idle_only_keeps_todays_triggers(engines, clock):
    engine = engines(cold_cache_causes="idle")
    _prompt_baseline(engine, clock)
    _switch_model(engine)
    engine.select_context(_with_prompt(SYSTEM_B, _messages()))
    assert engine.should_compress_preflight(_messages()) is False
    assert engine.should_compress(1000) is False
    clock.now = T0 + GAP
    assert engine.should_compress_preflight(_messages()) is True
    assert _event(engine)["cause"] == "idle"


def test_seconds_zero_turns_the_whole_family_off(engines, clock):
    engine = engines(cold_gap_compaction_seconds=0)
    _prompt_baseline(engine, clock)
    _switch_model(engine)
    engine.select_context(_with_prompt(SYSTEM_B, _messages()))
    clock.now = T0 + GAP
    assert engine.should_compress_preflight(_messages()) is False
    assert engine.should_compress(1000) is False
    assert _record(engine) == {}
