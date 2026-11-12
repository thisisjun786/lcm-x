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
from hermes_lcm.compaction_request import ROUTE_CALIBRATION_FIELD
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


def test_both_attempts_free_nothing_keeps_the_replay_redaction(engines, clock):
    engine = engines(sensitive_patterns_enabled=True, sensitive_patterns=["password_assignment"])
    secret = "SYNTHaaaa1111"
    messages = _messages(rows=2)[1:]  # two fresh-tail rows, no system row: nothing to summarise or stub
    messages[0] = dict(messages[0], content=f"deploy note password={secret} " + str(messages[0]["content"]) * 3)
    redacted_tokens = count_messages_tokens(engine._redact_active_replay_messages(messages))
    engine.threshold_tokens = int(redacted_tokens / 0.97)  # the cold band is [floor, threshold) on the redacted size
    _provider_call(engine, clock, T0)
    clock.now = T0 + GAP
    assert engine.should_compress_preflight(messages) is True
    result = engine.compress(messages, current_tokens=count_messages_tokens(messages))
    event = _event(engine)
    assert (event["attempts"], event["outcome"]) == (2, "no_progress")
    assert secret not in json.dumps(result)
    assert secret in json.dumps(messages)   # the caller's own list is not mutated


SECRET = "SYNTHbbbb2222"


def _tool_call_turn(call_id: str) -> list[dict]:
    """An assistant tool call whose JSON arguments carry a synthetic password, and its result."""
    arguments = json.dumps({"command": f"login --user ops password={SECRET}"})
    return [{"role": "assistant", "content": "running tool",
             "tool_calls": [{"id": call_id, "type": "function",
                             "function": {"name": "shell", "arguments": arguments}}]},
            {"role": "tool", "tool_call_id": call_id, "content": "logged in"}]


def test_a_shrinking_cold_pass_keeps_tool_call_redaction(engines, clock):
    # The first-attempt path: the pass shrinks and the returned replay keeps the tail's tool call, redacted.
    engine = engines(sensitive_patterns_enabled=True, sensitive_patterns=["password_assignment"])
    messages = _messages() + _tool_call_turn("call-tail")
    _provider_call(engine, clock, T0)
    clock.now = T0 + GAP
    assert engine.should_compress_preflight(messages) is True
    result = engine.compress(messages, current_tokens=count_messages_tokens(messages))
    assert (_event(engine)["attempts"], _event(engine)["outcome"]) == (1, "shrunk")
    assert any(m.get("tool_calls") for m in result)
    assert SECRET not in json.dumps(result)
    assert SECRET in json.dumps(messages)


@pytest.mark.parametrize("attempt", ["same_rows", "added_row", "not_a_list"])
def test_both_empty_attempts_return_the_redacted_host_rows(engines, clock, monkeypatch, caplog, attempt):
    # Whatever the empty attempts returned, the host gets its own rows, row for row, redacted (tool_calls too).
    engine = engines(sensitive_patterns_enabled=True, sensitive_patterns=["password_assignment"])
    messages = _messages(rows=4) + _tool_call_turn("call-0")
    tokens = count_messages_tokens(messages)
    redacted_tokens = count_messages_tokens(engine._redact_active_replay_messages(messages))
    engine.threshold_tokens = int(redacted_tokens / 0.97)  # the cold band is measured on the redacted size
    _provider_call(engine, clock, T0)
    clock.now = T0 + GAP
    assert engine.should_compress_preflight(messages) is True
    assert engine._compaction_request.cold_gap is True
    shapes = {"same_rows": lambda: [dict(m) for m in messages],
              "added_row": lambda: [*messages, {"role": "assistant", "content": "extra"}],
              "not_a_list": lambda: None}

    def empty_attempt(*args, **kwargs):
        engine._last_compression_status = "noop"
        return shapes[attempt]()

    monkeypatch.setattr(engine, "_compress_impl", empty_attempt)
    with caplog.at_level(logging.WARNING):
        result = engine.compress(messages, current_tokens=tokens)
    assert [r.getMessage().startswith(FREED_NOTHING) for r in caplog.records].count(True) == 1
    assert (_event(engine)["attempts"], _event(engine)["outcome"]) == (2, "no_progress")
    assert len(result) == len(messages) and [m["role"] for m in result] == [m["role"] for m in messages]
    assert SECRET not in json.dumps(result) and SECRET in json.dumps(messages)
    assert result[-2]["tool_calls"][0]["function"]["name"] == "shell"
    assert [m.get("content") for m in result[:-2]] == [m.get("content") for m in messages[:-2]]
    assert engine._ingest_cursor == len(result)


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


# -- 11. the deployment threshold (context_threshold 0.5) next to 0.7 -------------------------------------------------


WINDOW = 1_000_000
BAND_SHARE = 0.97  # host tokens just under the threshold, inside the cold band [floor, threshold)


@pytest.fixture
def deployed(tmp_path, monkeypatch):
    """The engine as a deployment builds it: ``context_threshold`` in its config and ``threshold_tokens`` from
    ``update_model`` on a 1M window (never assigned afterwards)."""
    built = []

    def build(threshold, name="d", **overrides):
        settings = dict(fresh_tail_count=2, leaf_chunk_tokens=100, threshold_full_sweep_enabled=True,
                        cold_gap_compaction_seconds=GAP, context_threshold=threshold,
                        database_path=str(tmp_path / name / "lcm.db"))
        settings.update(overrides)
        engine = LCMEngine(config=LCMConfig(**settings), hermes_home=str(tmp_path / name / "hermes_home"))
        built.append(engine)
        engine.on_session_start(CONVERSATION)
        engine.update_model("model-a", WINDOW, provider="provider-a", api_mode="responses")
        assert engine.threshold_tokens == int(WINDOW * threshold)

        def fake_leaf(chunk, focus_topic=None, deadline=None):
            del focus_topic, deadline
            return chunk, count_messages_tokens(chunk), "bounded", 1, 0

        monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", fake_leaf)
        return engine

    yield build
    for engine in built:
        engine.shutdown()


def _make_cold(engine, clock, cause, prompt_tokens):
    """The last provider call (``prompt_tokens``, sent with prompt A), then what makes the next call cold."""
    assert engine.select_context(_with_prompt(SYSTEM_A, _messages())) is None
    _provider_call(engine, clock, T0, prompt_tokens=prompt_tokens)
    clock.now = T0 + (GAP if cause in ("idle", "all") else 5)
    if cause in ("model_change", "all"):
        engine.update_model("model-b", WINDOW, provider="provider-a", api_mode="responses")
    if cause in ("prompt_change", "all"):
        assert engine.select_context(_with_prompt(SYSTEM_B, _messages())) is None


def _band_tool_conversation(tokens: int) -> list[dict]:
    """About ``tokens`` of aged-stub sized tool outputs (above the 2k aged tier, below the 10k first-sight tier)."""
    messages = [{"role": "system", "content": "system"}]
    payload = 8_000
    for turn in range(tokens // (payload + 80)):
        messages.append({"role": "user", "content": f"ask {turn} " + "word " * 20})
        messages += tool_pair(f"call-{turn}", payload_of(payload, word=f"w{turn % 10}"))
        messages.append({"role": "assistant", "content": f"reply {turn} " + "word " * 20})
    return messages


@pytest.mark.parametrize("threshold", [0.7, 0.5])
@pytest.mark.parametrize("cause", ["idle", "model_change", "prompt_change"])
def test_each_cause_requests_the_cold_pass_just_under_the_threshold(deployed, clock, caplog, threshold, cause):
    engine = deployed(threshold)
    host = int(engine.threshold_tokens * BAND_SHARE)
    assert engine._config.cold_gap_min_prompt_tokens <= host < engine.threshold_tokens
    _make_cold(engine, clock, cause, host)
    assert engine.threshold_tokens == int(WINDOW * threshold)  # a model switch recomputes the same threshold
    with caplog.at_level(logging.INFO):
        assert engine.should_compress(host) is True
    assert engine._compaction_request.cold_gap is True
    lines = _requested_lines(caplog)
    assert len(lines) == 1 and f"threshold={engine.threshold_tokens}" in lines[0], lines
    assert _event(engine)["cause"] == cause and _event(engine)["observed_prompt_tokens"] == host


@pytest.mark.parametrize("threshold", [0.7, 0.5])
def test_at_the_deployed_threshold_the_ordinary_path_runs(deployed, clock, threshold):
    engine = deployed(threshold)
    _make_cold(engine, clock, "all", engine.threshold_tokens)
    assert engine.should_compress(engine.threshold_tokens) is True
    assert engine._compaction_request.cold_gap is False and _record(engine) == {}


@pytest.mark.parametrize("threshold", [0.7, 0.5])
def test_below_the_floor_no_cause_ever_asks(deployed, clock, threshold):
    engine = deployed(threshold)
    below = engine._config.cold_gap_min_prompt_tokens - 1
    _make_cold(engine, clock, "all", below)  # idle, model and prompt all changed
    assert engine.should_compress(below) is False
    assert engine.should_compress_preflight(_messages()) is False
    assert engine._compaction_request.cold_gap is False and _record(engine) == {}


@pytest.mark.parametrize("threshold", [0.7, 0.5])
@pytest.mark.parametrize("ratio", [0.0, 0.2])
def test_a_band_pass_that_frees_nothing_is_retried_through_the_free_cuts(deployed, clock, monkeypatch, threshold,
                                                                        ratio):
    engine = deployed(threshold, leaf_chunk_tokens=1_000, stub_first_exit_target_ratio=ratio)
    messages = _band_tool_conversation(int(engine.threshold_tokens * BAND_SHARE))
    host = count_messages_tokens(messages)
    assert 0.95 * engine.threshold_tokens <= host < engine.threshold_tokens
    calls = _out_of_time(engine, monkeypatch)
    _make_cold(engine, clock, "idle", host)
    assert engine.should_compress_preflight(messages) is True  # the turn-start gate ingests the host rows
    stored_before = _store_digest(engine)
    result = engine.compress(messages, current_tokens=host)
    assert calls  # the first attempt tried a leaf and ran out of time
    event = _event(engine)
    assert (event["attempts"], event["outcome"], event["stop_reason"]) == (2, "shrunk", "stub_first_exit"), event
    # The free-cut target: min(threshold - leaf chunk, 0.95 x threshold, ratio x threshold, 0.95 x host).
    target = min(engine.threshold_tokens - 1_000, int(0.95 * engine.threshold_tokens), int(0.95 * host))
    if ratio:
        target = min(target, int(ratio * engine.threshold_tokens))
    assert count_messages_tokens(result) <= target
    assert [m["role"] for m in result] == [m["role"] for m in messages]  # stubs only, no row dropped
    assert _store_digest(engine) == stored_before


@pytest.mark.parametrize("threshold", [0.7, 0.5])
def test_a_band_retry_past_the_free_cuts_runs_the_leaf_loop(deployed, clock, monkeypatch, threshold):
    # Plain text rows: the free cuts cannot reach the 0.2 ratio target, so the retry goes on to the leaf loop.
    engine = deployed(threshold, leaf_chunk_tokens=20_000, stub_first_exit_target_ratio=0.2)
    messages = [{"role": "system", "content": "system"}]
    for index in range(int(engine.threshold_tokens * BAND_SHARE) // 8_000):
        messages.append({"role": "user" if index % 2 == 0 else "assistant",
                         "content": f"band-{index} " + payload_of(8_000, word=f"t{index % 10}")})
    host = count_messages_tokens(messages)
    assert 0.95 * engine.threshold_tokens <= host < engine.threshold_tokens
    retry, calls = [], []
    prepare = engine._prepare_cold_threshold_retry

    def armed():
        retry.append(True)
        prepare()

    def leaf(chunk, focus_topic=None, deadline=None):
        del focus_topic, deadline
        calls.append(bool(retry))
        if not retry:
            raise SweepBudgetExhausted("time_budget_exhausted")
        return chunk, count_messages_tokens(chunk), "bounded", 1, 0

    monkeypatch.setattr(engine, "_prepare_cold_threshold_retry", armed)
    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", leaf)
    _make_cold(engine, clock, "idle", host)
    assert engine.should_compress_preflight(messages) is True  # the turn-start gate ingests the host rows
    stored_before = _store_digest(engine)
    result = engine.compress(messages, current_tokens=host)
    assert False in calls and True in calls  # a leaf in each attempt: the retry reached the leaf loop
    event = _event(engine)
    assert (event["attempts"], event["outcome"]) == (2, "shrunk") and event["stop_reason"] != "stub_first_exit"
    assert count_messages_tokens(result) < 0.95 * host
    assert _store_digest(engine) == stored_before
    rows = {row["content"]: row["store_id"] for row in engine._store.get_session_messages(engine._session_id)}
    covered = {sid for node in engine._dag.get_session_nodes(engine._session_id) for sid in node.source_ids}
    kept = {m.get("content") for m in result}
    removed = [m for m in messages[1:] if m["content"] not in kept]
    assert removed and all(rows[m["content"]] in covered for m in removed)


@pytest.mark.parametrize("threshold", [0.7, 0.5])
def test_a_band_pass_where_both_attempts_free_nothing_keeps_the_list(deployed, clock, caplog, threshold):
    engine = deployed(threshold)
    messages = _messages(rows=2)  # every row is in the fresh tail
    host = int(engine.threshold_tokens * BAND_SHARE)
    _make_cold(engine, clock, "idle", host)
    assert engine.should_compress_preflight(messages) is True  # the turn-start gate ingests the host rows
    stored_before = _store_digest(engine)
    with caplog.at_level(logging.INFO):
        result = engine.compress(messages, current_tokens=host)
    assert result == messages
    warnings = [r for r in caplog.records if r.getMessage().startswith(FREED_NOTHING)]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING
    assert (_event(engine)["attempts"], _event(engine)["outcome"]) == (2, "no_progress")
    assert _store_digest(engine) == stored_before
    assert engine.should_compress(host) is False  # consumed: the host's next iteration sends, no loop


# -- 12. the floor after a model or provider flip: a ratio learned from this conversation's own calls -------------------


A, B = ("model-a", "provider-a"), ("model-b", "provider-a")


def _route(engine, model, provider):
    engine.update_model(model, WINDOW, provider=provider, api_mode="responses")


def _call(engine, clock, at, route, request, reported, *, observe=True):
    """One provider call on ``route``: select_context sees ``request``, the provider reports ``reported``."""
    _route(engine, *route)
    if observe:
        assert engine.select_context(request) is None
    _provider_call(engine, clock, at, prompt_tokens=reported)
    return count_messages_tokens(request)


def _calibration(engine) -> list:
    return (engine._store.read_compaction_telemetry(CONVERSATION) or {}).get(ROUTE_CALIBRATION_FIELD)


def _seed(engine, clock, *, last_reported, last_request, ratio_factor):
    """An earlier call on B, then the last call on A (``last_reported`` for ``last_request``); B's reported/counted
    ratio is ``ratio_factor`` times A's, the shape the deployment measured (x1.51 to x1.62)."""
    a_ratio = last_reported / count_messages_tokens(last_request)
    b_request = _band_tool_conversation(150_000)
    b_counted = count_messages_tokens(b_request)
    _call(engine, clock, T0 - 600, B, b_request, int(b_counted * a_ratio * ratio_factor))
    _call(engine, clock, T0, A, last_request, last_reported)
    reported, counted = _calibration(engine)[0][1:]
    return reported / counted


def test_a_flip_estimates_the_next_routes_prompt_from_its_learned_ratio(deployed, clock, caplog):
    # 11-10 21:27 shape: last call on A reported 268k; the same history on B is ~1.6x by B's own count. The engine's
    # count of the new request and the last observed figure are both under the 300k floor: only the ratio sees it.
    engine = deployed(0.5)
    b_ratio = _seed(engine, clock, last_reported=268_000, last_request=_band_tool_conversation(260_000),
                    ratio_factor=1.62)
    clock.now = T0 + 20
    _route(engine, *B)
    messages = _band_tool_conversation(265_000)
    counted_now = count_messages_tokens(messages)
    floor = engine._config.cold_gap_min_prompt_tokens
    assert counted_now < floor and 268_000 < floor and floor <= counted_now * b_ratio < engine.threshold_tokens
    with caplog.at_level(logging.INFO):
        assert engine.should_compress_preflight(messages) is True
    event = _event(engine)
    assert (event["cause"], event["observed_prompt_tokens"]) == ("model_change", 268_000)
    assert abs(event["floor_tokens"] - counted_now * b_ratio) <= 0.02 * counted_now * b_ratio
    assert [f"floor_tokens={event['floor_tokens']} cause=model_change" in line
            for line in _requested_lines(caplog)] == [True]
    result = engine.compress(messages, current_tokens=counted_now)
    assert _event(engine)["outcome"] == "shrunk" and count_messages_tokens(result) < 0.95 * counted_now


def test_a_flip_after_an_idle_gap_uses_the_same_estimate(deployed, clock):
    # 11-08 20:10 shape: last call on A 233k, idle 23,178 s, the next route's estimate ~369k.
    engine = deployed(0.5)
    b_ratio = _seed(engine, clock, last_reported=233_000, last_request=_band_tool_conversation(226_000),
                    ratio_factor=1.58)
    clock.now = T0 + 23_178
    _route(engine, *B)
    messages = _band_tool_conversation(230_000)
    assert count_messages_tokens(messages) * b_ratio >= engine._config.cold_gap_min_prompt_tokens
    assert engine.should_compress_preflight(messages) is True
    event = _event(engine)
    assert (event["cause"], event["observed_prompt_tokens"]) == ("idle", 233_000)
    assert event["floor_tokens"] >= engine._config.cold_gap_min_prompt_tokens


def test_a_flip_under_the_floor_on_both_figures_does_not_ask(deployed, clock):
    engine = deployed(0.5)
    b_ratio = _seed(engine, clock, last_reported=268_000, last_request=_band_tool_conversation(260_000),
                    ratio_factor=1.62)
    clock.now = T0 + GAP  # idle and model change both due
    _route(engine, *B)
    messages = _band_tool_conversation(150_000)
    assert count_messages_tokens(messages) * b_ratio < engine._config.cold_gap_min_prompt_tokens
    assert engine.should_compress_preflight(messages) is False
    assert engine._compaction_request.cold_gap is False and _record(engine) == {}


def test_a_route_this_conversation_never_called_keeps_the_last_observed_floor(deployed, clock):
    # No ratio for the next route: exactly 687dbc6b (the last call's figure decides), even with a large request.
    engine = deployed(0.5)
    _seed(engine, clock, last_reported=268_000, last_request=_band_tool_conversation(260_000), ratio_factor=1.62)
    clock.now = T0 + GAP
    _route(engine, "model-c", "provider-c")
    assert engine.should_compress_preflight(_band_tool_conversation(265_000)) is False
    assert _record(engine) == {}


@pytest.mark.parametrize("gap", [5, GAP])
def test_the_same_route_keeps_the_provider_reported_floor(deployed, clock, gap):
    engine = deployed(0.5)
    _seed(engine, clock, last_reported=268_000, last_request=_band_tool_conversation(260_000), ratio_factor=1.62)
    clock.now = T0 + gap
    messages = _band_tool_conversation(430_000)
    assert count_messages_tokens(messages) >= engine._config.cold_gap_min_prompt_tokens
    assert engine.should_compress_preflight(messages) is False
    assert engine.should_compress(count_messages_tokens(messages)) is False
    assert _record(engine) == {}


def test_a_compaction_after_the_last_call_still_suppresses_the_flip(deployed, clock):
    engine = deployed(0.5)
    _seed(engine, clock, last_reported=268_000, last_request=_band_tool_conversation(260_000), ratio_factor=1.62)
    small = _messages()
    engine.compress(small, current_tokens=count_messages_tokens(small), force=True)  # the engine compacted
    assert engine.compression_count >= 1
    clock.now = T0 + 20
    _route(engine, *B)
    assert engine.should_compress_preflight(_band_tool_conversation(265_000)) is False
    assert _record(engine) == {}


@pytest.mark.parametrize("observed", [True, False], ids=["after_select_context", "before_select_context"])
def test_the_gate_estimates_from_select_contexts_count(deployed, clock, observed):
    # The host's gate figure is anchored on the last call's usage (A's count plus a delta estimate).
    engine = deployed(0.5)
    b_ratio = _seed(engine, clock, last_reported=268_000, last_request=_band_tool_conversation(260_000),
                    ratio_factor=1.62)
    clock.now = T0 + 20
    _route(engine, *B)
    request = _band_tool_conversation(265_000)
    anchored = 268_000 + 2_000
    if observed:  # the per-request hook runs right before the gate
        assert engine.select_context(request) is None
        assert engine.should_compress(anchored) is True
        event = _event(engine)
        assert event["cause"] == "model_change"
        assert abs(event["floor_tokens"] - count_messages_tokens(request) * b_ratio) <= 1
    else:  # a gate call before this request's select_context (the host's turn-start gate): the anchored figure only
        assert engine.should_compress(anchored) is False
        assert _record(engine) == {}


def test_each_answer_records_its_routes_pair_with_the_call(deployed, clock, monkeypatch):
    engine = deployed(0.5)
    writes = []
    original = engine._store.increment_compaction_telemetry

    def spy(conversation_id, increment, updates):
        writes.append(dict(updates))
        return original(conversation_id, increment, updates)

    monkeypatch.setattr(engine._store, "increment_compaction_telemetry", spy)
    small = _messages()
    counted = _call(engine, clock, T0, A, small, 9_000)
    assert len(writes) == 1 and writes[0]["last_api_call_at"] == T0  # one write carries both
    assert writes[0][ROUTE_CALIBRATION_FIELD] == [["provider-a/model-a", 9_000, counted]]
    _call(engine, clock, T0 + 1, B, small, 14_000)
    _call(engine, clock, T0 + 2, A, small + _messages(rows=2)[1:], 9_500)
    entries = _calibration(engine)
    assert [entry[0] for entry in entries] == ["provider-a/model-b", "provider-a/model-a"]  # B survives, A latest
    assert entries[1][1] == 9_500
    _call(engine, clock, T0 + 3, B, small, 15_000, observe=False)  # no engine count: B's pair is kept as it was
    assert _calibration(engine)[0] == ["provider-a/model-b", 14_000, counted]


def test_the_calibration_keeps_the_eight_latest_routes(deployed, clock):
    engine = deployed(0.5)
    for index in range(10):
        _call(engine, clock, T0 + index, (f"model-{index}", "provider-a"), _messages(), 9_000 + index)
    entries = _calibration(engine)
    assert [entry[0] for entry in entries] == [f"provider-a/model-{index}" for index in range(2, 10)]


@pytest.mark.parametrize("raw", [None, "garbage", [["k", 1]], [["provider-a/model-b", "x", 1], 7]])
def test_an_old_or_malformed_calibration_reads_empty(deployed, clock, raw):
    engine = deployed(0.5)
    _seed(engine, clock, last_reported=268_000, last_request=_band_tool_conversation(260_000), ratio_factor=1.62)
    telemetry = engine._store.read_compaction_telemetry(CONVERSATION)
    if raw is None:
        telemetry.pop(ROUTE_CALIBRATION_FIELD)
    else:
        telemetry[ROUTE_CALIBRATION_FIELD] = raw
    engine._store.write_compaction_telemetry(CONVERSATION, telemetry)
    clock.now = T0 + 20
    _route(engine, *B)
    assert engine.should_compress_preflight(_band_tool_conversation(265_000)) is False  # as 687dbc6b
    assert _record(engine) == {}
    _call(engine, clock, T0 + 30, B, _messages(), 9_000)
    assert [entry[0] for entry in _calibration(engine)] == ["provider-a/model-b"]


# -- 13. rows fell but the tokens did not: unpaired old rows the leaf loop must keep live ------------------------------


STOP_LINE = "LCM compaction stop:"


def _conv_event(engine) -> dict:
    record = engine._store.read_metadata_json(engine._cold_gap_metadata_key(engine._conversation_id)) or {}
    return record.get("last_event") or {}


def _unpaired_backlog_case(tmp_path, monkeypatch, clock, caplog, old_turns):
    """A host view of ``old_turns`` old turns whose rows have no stored, paired copy (``_uid_leaf_candidates`` keeps
    unresolved occurrences live, so no leaf takes them) and 3 newer stored, uid-bound turns. The first attempt
    folds the newer rows into one leaf, finds no candidate left, ends ``raw_prefix_drained``, and the post-drain
    condensation's ``summary_prefix_target_reached`` becomes the sweep's stop reason."""
    import itertools

    from tests.test_fork_covered_head import _rows, _store
    from tests.test_fork_late_copy_mapping import _home

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **kw: ("Synthetic leaf summary.", 1))
    engine = LCMEngine(config=LCMConfig(fresh_tail_count=4, leaf_chunk_tokens=3000, dynamic_leaf_chunk_max=3000,
                                        threshold_full_sweep_enabled=True, cold_gap_compaction_seconds=GAP,
                                        database_path=str(tmp_path / "lcm.db")), hermes_home=_home(tmp_path))
    engine.on_session_start("S", platform="telegram", context_length=WINDOW, conversation_id="conv")  # _store's
    engine.update_model("model-a", WINDOW, provider="provider-a", api_mode="responses")
    _provider_call(engine, clock, T0)  # the last provider call, an idle gap before this turn
    clock.now = T0 + GAP
    stamps = itertools.count(1_700_000_000)
    old, new = _rows("old", old_turns, stamps), _rows("new", 3, stamps)
    _store(engine, new)
    view = [{"role": "system", "content": "sys"}, *[dict(row) for row in old + new]]
    engine._ingest_cursor = len(view)  # the host replays rows the store already took, no new ingest
    tokens = count_messages_tokens(view)
    engine.threshold_tokens = int(tokens / 0.97)
    assert engine.should_compress(tokens) is True and engine._compaction_request.cold_gap is True
    stored = len(engine._store.get_session_messages("S"))
    with caplog.at_level(logging.INFO):
        result = engine.compress(view, current_tokens=tokens)
    stop = [r.getMessage() for r in caplog.records if r.getMessage().startswith(STOP_LINE)]
    return engine, view, result, stop, stored


def test_a_pass_that_drops_rows_but_frees_under_five_percent_is_retried_and_reported(tmp_path, monkeypatch,
                                                                                    clock, caplog):
    engine, view, result, stop, stored = _unpaired_backlog_case(tmp_path, monkeypatch, clock, caplog, 40)
    try:
        retry = [r for r in caplog.records if r.getMessage().startswith(
            "LCM cold-cache compaction freed nothing; retrying through the threshold path")]
        assert len(retry) == 1 and "stop=raw_prefix_drained" in retry[0].getMessage()
        warnings = [r for r in caplog.records if r.getMessage().startswith(FREED_NOTHING)]
        assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING
        event = _conv_event(engine)
        assert (event["attempts"], event["outcome"]) == (2, "no_progress")
        assert event["leaf_stop_reason"] == "raw_prefix_drained"
        assert result == view  # the host's own rows (nothing to redact), never a 4%-smaller list counted as done
        # One stop line per compress(): the retry's, whose sweep ran no post-drain condensation to relabel it.
        assert len(stop) == 1 and "reason=raw_prefix_drained" in stop[0] and "unpaired_count=160" in stop[0]
        assert len(engine._store.get_session_messages("S")) == stored  # nothing deleted
    finally:
        engine.shutdown()


def test_the_same_shape_under_five_percent_also_on_the_first_attempt(tmp_path, monkeypatch, clock, caplog):
    # The 11-11 shape in numbers: rows fall (a leaf took the newer rows) while the tokens fall by under 5%.
    engine = None
    original = LCMEngine._compress_impl
    first = {}

    def spy(self, messages, **kwargs):
        out = original(self, messages, **kwargs)
        first.setdefault("rows", (len(messages), len(out)))
        first.setdefault("tokens", (count_messages_tokens(messages), count_messages_tokens(out)))
        return out

    monkeypatch.setattr(LCMEngine, "_compress_impl", spy)
    engine, view, result, stop, stored = _unpaired_backlog_case(tmp_path, monkeypatch, clock, caplog, 40)
    try:
        rows_before, rows_after = first["rows"]
        tokens_before, tokens_after = first["tokens"]
        assert rows_after < rows_before and 0 < tokens_before - tokens_after < 0.05 * tokens_before
    finally:
        engine.shutdown()


def test_a_pass_that_frees_five_percent_or_more_is_shrunk_without_a_retry(tmp_path, monkeypatch, clock, caplog):
    engine, view, result, stop, stored = _unpaired_backlog_case(tmp_path, monkeypatch, clock, caplog, 5)
    try:
        assert not any("retrying through the threshold path" in r.getMessage() for r in caplog.records)
        assert not any(r.getMessage().startswith(FREED_NOTHING) for r in caplog.records)
        event = _conv_event(engine)
        assert (event["attempts"], event["outcome"]) == (1, "shrunk")
        assert event["leaf_stop_reason"] == "raw_prefix_drained"
        assert event["stop_reason"] == "summary_prefix_target_reached"  # the post-drain condensation's result
        assert len(stop) == 1 and "reason=summary_prefix_target_reached" in stop[0]
        assert " leaf_stop=raw_prefix_drained" in stop[0]
        assert count_messages_tokens(result) <= 0.95 * count_messages_tokens(view)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("host_share, expected", [(None, False), (1.0, False), (1.1, True)])
def test_freed_nothing_is_judged_on_the_hosts_request_tokens(host_share, expected):
    # Exactly 5% of the list freed is progress; the same cut is under 5% of a host request 10% larger.
    messages = [{"role": "user", "content": "x" * 4_000}]
    before = count_messages_tokens(messages)
    result = [{"role": "user", "content": "x" * 4_000}]
    while before - count_messages_tokens(result) < before * 0.05:
        result[0]["content"] = result[0]["content"][:-4]
    assert before - count_messages_tokens(result) <= before * 0.05 + 1
    host = None if host_share is None else int(before * host_share)
    freed = before - count_messages_tokens(result)
    assert LCMEngine._cold_pass_freed_nothing(messages, result, host) is (
        expected or max(before, host or 0) - freed > max(before, host or 0) * 0.95)
    assert LCMEngine._cold_pass_freed_nothing(messages, result, host) is expected
    assert LCMEngine._cold_pass_freed_nothing(messages, None, host) is True


def test_seconds_zero_turns_the_whole_family_off(engines, clock):
    engine = engines(cold_gap_compaction_seconds=0)
    _prompt_baseline(engine, clock)
    _switch_model(engine)
    engine.select_context(_with_prompt(SYSTEM_B, _messages()))
    clock.now = T0 + GAP
    assert engine.should_compress_preflight(_messages()) is False
    assert engine.should_compress(1000) is False
    assert _record(engine) == {}


# -- 15. a stored duplicate of a covered row blocks the frontier (11-12 hub copy: store 100501 = 100464) -------------


def _twin_engine(tmp_path, monkeypatch):
    from tests.test_fork_late_copy_mapping import _home

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **kw: ("Synthetic leaf summary.", 1))
    engine = LCMEngine(config=LCMConfig(fresh_tail_count=4, leaf_chunk_tokens=3000, dynamic_leaf_chunk_max=3000,
                                        threshold_full_sweep_enabled=True, cold_gap_compaction_seconds=GAP,
                                        database_path=str(tmp_path / "lcm.db")), hermes_home=_home(tmp_path))
    engine.on_session_start("S", platform="telegram", context_length=WINDOW, conversation_id="conv")
    engine.update_model("model-a", WINDOW, provider="provider-a", api_mode="responses")
    return engine


def _twin_frontier(engine) -> int:
    return engine._lifecycle.get_by_conversation("conv").current_frontier_store_id


def _publish_cover(engine, rows_ids, session="S"):
    """One depth-0 leaf over ``rows_ids`` (host row, store id), published as a sparse uid leaf is."""
    from hermes_lcm.dag import SummaryNode

    engine._uid_source_bindings = {sid: row["message_uid"] for row, sid in rows_ids if row.get("message_uid")}
    engine._uid_sparse_coverage = True
    engine._store_complete_copy_exclusion_proofs = {}
    ids = [sid for _row, sid in rows_ids]
    node = SummaryNode(session_id=session, summary="Synthetic leaf summary.", depth=0, source_ids=ids,
                       source_type="messages", token_count=8, source_token_count=5000)
    engine._publish_leaf_node(node, level=1, model="synthetic",
                              expected_frontier=engine._lifecycle.get_by_conversation("conv").current_frontier_store_id,
                              consumed_store_ids=ids, publication_excluded_store_ids=[], filter_exclusion_proofs={},
                              carried_ranges=[])
    engine._uid_sparse_coverage = False


def _twin_session(engine, stamps, shape):
    """30 stored, uid-bound turns; right after the tool result T of turn 3 a second, uid-less tool row T' is stored
    (``identical``: T's bytes, as store 100501 repeats 100464). Leaves of 20 rows cover every row but T', so the
    frontier stands at T, in front of T'. The adversarial shapes change T' (one character, another tool_call_id)
    or move its only twin to another session's leaf."""
    from tests.test_fork_covered_head import _rows, _store

    rows = _rows("old", 30, stamps)
    hole = next(i for i, row in enumerate(rows) if row["role"] == "tool" and row["content"].startswith("old output 3 "))
    twin = {k: v for k, v in rows[hole].items() if k != "message_uid"}
    if shape in ("one_character", "twin_in_other_session"):
        twin["content"] = twin["content"][:-1] + "X"
    elif shape == "other_call_id":
        twin["tool_call_id"] = "old_c3_again"
    ids = _store(engine, rows[:hole + 1])
    copy_sid = engine._store.append("S", twin, conversation_id="conv")
    ids += _store(engine, rows[hole + 1:])
    pairs = [(row, sid) for row, sid in zip(rows, ids)]
    if shape == "twin_in_other_session":
        # T' differs from T in this session; its byte-identical twin is another session's row, which that
        # session's leaf covers.
        from hermes_lcm.dag import SummaryNode

        other = engine._store.append("OTHER", dict(twin), conversation_id="conv")
        engine._dag.add_node(SummaryNode(session_id="OTHER", summary="Other session leaf.", depth=0,
                                         source_ids=[other], source_type="messages", token_count=8,
                                         source_token_count=100))
    uncovered = copy_sid
    covered_pairs = [pair for pair in pairs if pair[1] != uncovered]
    for start in range(0, len(covered_pairs), 20):
        _publish_cover(engine, covered_pairs[start:start + 20])
    return rows, ids, hole, copy_sid, uncovered


def _cold_pass_over(engine, clock, caplog, view):
    engine._ingest_cursor = len(view)
    tokens = count_messages_tokens(view)
    engine.threshold_tokens = int(tokens / 0.97)
    _provider_call(engine, clock, T0)
    clock.now = T0 + GAP
    assert engine.should_compress(tokens) is True and engine._compaction_request.cold_gap is True
    with caplog.at_level(logging.INFO):
        result = engine.compress(view, current_tokens=tokens)
    return tokens, result


def test_a_covered_duplicate_no_longer_holds_the_frontier(tmp_path, monkeypatch, clock, caplog):
    import itertools

    from hermes_lcm import tools as lcm_tools
    from tests.test_fork_covered_head import _rows, _store

    engine = _twin_engine(tmp_path, monkeypatch)
    try:
        stamps = itertools.count(1_700_000_000)
        rows, ids, hole, copy_sid, _uncovered = _twin_session(engine, stamps, "identical")
        assert _twin_frontier(engine) == ids[hole] and copy_sid == ids[hole] + 1  # stuck in front of T'
        new = _rows("new", 2, stamps)
        _store(engine, new)
        view = [{"role": "system", "content": "sys"}, *[dict(row) for row in rows[hole + 1:] + new]]
        stored = len(engine._store.get_session_messages("S"))
        tokens, result = _cold_pass_over(engine, clock, caplog, view)
        # T' is accounted for by its covered twin: the frontier passes it and the covered rows behind it.
        assert _twin_frontier(engine) >= ids[-1]
        assert copy_sid not in engine._store_complete_node_covered({copy_sid})  # T' is not summarised twice
        assert count_messages_tokens(result) <= 0.5 * tokens  # the covered rows left as their summaries
        assert _conv_event(engine)["outcome"] == "shrunk"
        assert len(engine._store.get_session_messages("S")) == stored  # nothing deleted
        expanded = json.loads(lcm_tools.lcm_expand({"store_id": copy_sid, "max_tokens": 100_000}, engine=engine))
        assert rows[hole]["content"] in json.dumps(expanded, ensure_ascii=False)  # T' is still there, whole
    finally:
        engine.shutdown()


@pytest.mark.parametrize("shape", ["one_character", "other_call_id", "twin_in_other_session"])
def test_a_duplicate_without_a_covered_twin_of_its_own_still_holds_the_frontier(tmp_path, monkeypatch, clock, caplog,
                                                                               shape):
    import itertools

    from tests.test_fork_covered_head import _rows, _store

    engine = _twin_engine(tmp_path, monkeypatch)
    try:
        stamps = itertools.count(1_700_000_000)
        rows, ids, hole, copy_sid, uncovered = _twin_session(engine, stamps, shape)
        assert _twin_frontier(engine) == uncovered - 1
        new = _rows("new", 2, stamps)
        _store(engine, new)
        view = [{"role": "system", "content": "sys"}, *[dict(row) for row in rows[hole + 1:] + new]]
        _cold_pass_over(engine, clock, caplog, view)
        # Not a byte-identical copy of a row a leaf of this session covers: the frontier never passes it unsummarised
        # (a hidden-only leaf takes it as the distinct row it is).
        assert _twin_frontier(engine) < uncovered or uncovered in engine._store_complete_node_covered({uncovered})
    finally:
        engine.shutdown()


# -- 16. the host list skips covered stored rows, so the #457 exact run never matches (hub 11-10: 280 of 887) -------


def _skipped_rows_session(engine, stamps, *, frontier_past=True):
    """30 stored, uid-bound turns; after every 4th turn's reply the store also holds a uid-less row the host list
    never shows (an older stored copy). Leaves of 20 rows cover every stored row, so the frontier is at the last
    one, and the host run through F is not an exact run of the stored rows (the skipped rows sit inside it)."""
    from tests.test_fork_covered_head import _rows, _store

    rows = _rows("old", 30, stamps)
    pairs = []
    for index, row in enumerate(rows):
        pairs.append((row, _store(engine, [row])[0]))
        if row["role"] == "assistant" and not row.get("tool_calls") and index % 16 == 3:
            skipped = engine._store.append("S", {"role": "assistant", "content": "older stored reply " + str(index),
                                                 "timestamp": row["timestamp"]}, conversation_id="conv")
            pairs.append(({}, skipped))
    for start in range(0, len(pairs), 20):
        _publish_cover(engine, pairs[start:start + 20])
    return rows, [sid for row, sid in pairs if row]


def test_covered_host_rows_leave_as_their_summary_when_the_store_holds_rows_the_host_skips(tmp_path, monkeypatch,
                                                                                            clock, caplog):
    import itertools

    from hermes_lcm import tools as lcm_tools
    from tests.test_fork_covered_head import _rows, _store

    engine = _twin_engine(tmp_path, monkeypatch)
    try:
        stamps = itertools.count(1_700_000_000)
        rows, ids = _skipped_rows_session(engine, stamps)
        assert _twin_frontier(engine) == max(engine._store_complete_node_covered(set(ids)) | set(ids))
        new = _rows("new", 2, stamps)
        _store(engine, new)
        view = [{"role": "system", "content": "sys"}, *[dict(row) for row in rows + new]]
        stored = len(engine._store.get_session_messages("S"))
        tokens, result = _cold_pass_over(engine, clock, caplog, view)
        assert count_messages_tokens(result) <= 0.5 * tokens
        assert _conv_event(engine)["outcome"] == "shrunk"
        assert len(engine._store.get_session_messages("S")) == stored  # nothing deleted
        assembled = json.dumps(result, ensure_ascii=False)
        assert "Synthetic leaf summary." in assembled and rows[0]["content"] not in assembled
        # The summary that replaced the first host row is a leaf over that row, and the leaf expands to its text.
        node = next(n for n in engine._dag.get_session_nodes("S") if n.depth == 0 and ids[0] in n.source_ids)
        expanded = json.loads(lcm_tools.lcm_expand({"node_id": node.node_id, "max_tokens": 100_000}, engine=engine))
        assert rows[0]["content"] in json.dumps(expanded, ensure_ascii=False)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("shape", ["no_uid", "above_frontier", "uncovered", "mixed_versions", "other_session"])
def test_a_host_row_the_uid_drop_cannot_prove_stays_in_the_list(tmp_path, monkeypatch, shape):
    import itertools

    from hermes_lcm.dag import SummaryNode
    from tests.test_fork_rewritten_uid_mapping import bind

    engine = _twin_engine(tmp_path, monkeypatch)
    try:
        stamps = itertools.count(1_700_000_000)
        rows, ids = _skipped_rows_session(engine, stamps)
        view = [{"role": "system", "content": "sys"}, *[dict(row) for row in rows]]
        k = next(i for i, row in enumerate(view) if row.get("role") == "user" and i > 40)  # mid-run, a turn start
        uid = view[k]["message_uid"]
        conn = engine._store._conn
        if shape == "no_uid":
            view[k].pop("message_uid")
        elif shape == "above_frontier":
            later = engine._store.append("S", {"role": "user", "content": "later copy"}, conversation_id="conv")
            engine._dag.add_node(SummaryNode(session_id="S", summary="Sparse leaf above F.", depth=0,
                                             source_ids=[later], source_type="messages", token_count=8,
                                             source_token_count=100))
            bind(engine, uid, later, kind="version")
        elif shape in ("uncovered", "mixed_versions"):
            loose = engine._store.append("S", {"role": "user", "content": "uncovered version"}, conversation_id="conv")
            bind(engine, uid, loose, kind="version")
            if shape == "uncovered":  # the row's only binding is the uncovered one
                conn.execute("DELETE FROM host_uid_bindings WHERE uid = ? AND store_id != ?", (uid, loose))
            frontier = loose  # at or below F, as a frontier moved by an exclusion would leave it
            conn.execute("UPDATE lcm_lifecycle_state SET current_frontier_store_id = ? WHERE conversation_id = ?",
                         (frontier, "conv"))
            conn.commit()
            engine._last_compacted_store_id = frontier
        elif shape == "other_session":
            other = engine._store.append("OTHER", {"role": "user", "content": view[k]["content"]}, conversation_id="conv")
            engine._dag.add_node(SummaryNode(session_id="OTHER", summary="Other session leaf.", depth=0,
                                             source_ids=[other], source_type="messages", token_count=8,
                                             source_token_count=100))
            conn.execute("DELETE FROM host_uid_bindings WHERE uid = ?", (uid,))
            conn.commit()
            bind(engine, uid, other)
        engine._last_compacted_store_id = _twin_frontier(engine)  # the bound engine's marker, as compress() keeps it
        drops, _ids, _tokens, _kept = engine._committed_replay_drops(view, 1)
        assert drops and k not in drops and max(drops) < k  # the rows before it go, it and every later row stay
    finally:
        engine.shutdown()


# -- 17. one cold pass finishes the work: the drop runs again after the pass moved the frontier -------------------


@pytest.mark.parametrize("shape", ["identical", "one_character"])
def test_one_cold_pass_replaces_the_covered_rows_its_own_retry_uncovered(tmp_path, monkeypatch, clock, caplog, shape):
    # identical: the retry moves the frontier over a covered-twin copy; one_character: the retry's hidden-only leaf
    # fills the hole and the frontier then moves over the covered rows. The first attempt is emptied (as the hub
    # replay's FIRST_EMPTY), so all of that happens inside the retry, one compress() call.
    import itertools

    from hermes_lcm import tools as lcm_tools
    from tests.test_fork_covered_head import _rows, _store

    engine = _twin_engine(tmp_path, monkeypatch)
    try:
        stamps = itertools.count(1_700_000_000)
        rows, ids, hole, copy_sid, _uncovered = _twin_session(engine, stamps, shape)
        assert _twin_frontier(engine) == ids[hole]
        new = _rows("new", 2, stamps)
        _store(engine, new)
        view = [{"role": "system", "content": "sys"}, *[dict(row) for row in rows[hole + 1:] + new]]
        original = LCMEngine._compress_impl
        calls = []

        def first_empty(self, messages, **kwargs):
            calls.append(len(messages))
            return list(messages) if len(calls) == 1 else original(self, messages, **kwargs)

        monkeypatch.setattr(LCMEngine, "_compress_impl", first_empty)
        stored = len(engine._store.get_session_messages("S"))
        tokens, result = _cold_pass_over(engine, clock, caplog, view)
        assert len(calls) == 2  # the emptied first attempt and the retry
        assert _twin_frontier(engine) >= ids[-1]
        assert count_messages_tokens(result) <= 0.5 * tokens
        assert _conv_event(engine)["outcome"] == "shrunk"
        assert len(engine._store.get_session_messages("S")) == stored  # nothing deleted
        replaced = rows[hole + 1]
        assert replaced["content"] not in json.dumps(result, ensure_ascii=False)
        node = next(n for n in engine._dag.get_session_nodes("S") if n.depth == 0 and ids[hole + 1] in n.source_ids)
        expanded = json.loads(lcm_tools.lcm_expand({"node_id": node.node_id, "max_tokens": 100_000}, engine=engine))
        assert replaced["content"] in json.dumps(expanded, ensure_ascii=False)
    finally:
        engine.shutdown()


def _count_replay_drops(monkeypatch) -> list:
    original = LCMEngine._committed_replay_drops
    calls = []

    def counted(self, working, start):
        calls.append(start)
        return original(self, working, start)

    monkeypatch.setattr(LCMEngine, "_committed_replay_drops", counted)
    return calls


def test_the_drop_is_not_run_again_when_the_pass_moved_nothing(tmp_path, monkeypatch, caplog):
    # A uid view, every host row covered at or below F, and the drop stops at the first host row (no uid): the pass
    # drains without a leaf and without moving the frontier, so the drop runs once, as before.
    import itertools

    engine = _twin_engine(tmp_path, monkeypatch)
    try:
        rows, ids = _skipped_rows_session(engine, itertools.count(1_700_000_000))
        engine._last_compacted_store_id = _twin_frontier(engine)
        view = [{"role": "system", "content": "sys"}, *[dict(row) for row in rows]]
        view[1].pop("message_uid")
        calls = _count_replay_drops(monkeypatch)
        engine.threshold_tokens = int(count_messages_tokens(view) / 1.2)
        with caplog.at_level(logging.INFO):
            engine.compress(view, current_tokens=count_messages_tokens(view), force=True)
        assert any("reason=raw_prefix_drained" in r.getMessage() or " leaf_stop=raw_prefix_drained" in r.getMessage()
                   for r in caplog.records if r.getMessage().startswith(STOP_LINE))  # the drained point was reached
        assert len(calls) == 1
    finally:
        engine.shutdown()


def test_the_drop_is_not_run_again_in_a_view_without_host_uids(tmp_path, monkeypatch, caplog):
    # The one_character shape with the host uids stripped from the view: the pass publishes the hole-filling leaf
    # (so it did move something), drains, and still runs the drop once, as the uid path never applies here.
    import itertools

    engine = _twin_engine(tmp_path, monkeypatch)
    try:
        rows, ids, hole, copy_sid, _uncovered = _twin_session(engine, itertools.count(1_700_000_000), "one_character")
        view = [{"role": "system", "content": "sys"},
                *[{k: v for k, v in row.items() if k != "message_uid"} for row in rows[hole + 1:]]]
        engine._ingest_cursor = len(view)
        calls = _count_replay_drops(monkeypatch)
        tokens = count_messages_tokens(view)
        engine.threshold_tokens = int(tokens / 1.2)
        with caplog.at_level(logging.INFO):
            engine.compress(view, current_tokens=tokens, force=True)
        assert copy_sid in engine._store_complete_node_covered({copy_sid})  # a leaf was published in this pass
        assert any("reason=raw_prefix_drained" in r.getMessage() or " leaf_stop=raw_prefix_drained" in r.getMessage()
                   for r in caplog.records if r.getMessage().startswith(STOP_LINE))
        assert len(calls) == 1
    finally:
        engine.shutdown()
