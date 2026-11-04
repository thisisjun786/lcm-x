"""Fork cold-gap compaction: below the threshold, after an idle gap of at least ``cold_gap_compaction_seconds`` since
this conversation's last provider call, the turn-start preflight asks for the threshold full sweep (once per gap).

Synthetic fixtures only; the gap comes from a fake ``time.time``, never a sleep.
"""

import json
import logging
import time

import pytest

from hermes_lcm import tokens as lcm_tokens
from hermes_lcm import tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens

GAP = 3600
THRESHOLD = 700_000
T0 = 1_800_000_000.0
CONVERSATION = "test-session"


class _Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def character_estimate(monkeypatch):
    monkeypatch.setattr(lcm_tokens, "_get_encoder", lambda: None)
    monkeypatch.setattr(lcm_tokens, "_encoder_generation", -1)


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock(T0)
    monkeypatch.setattr(time, "time", fake)
    return fake


def _engine(tmp_path, monkeypatch, *, seconds=GAP, summary="bounded", **overrides) -> LCMEngine:
    engine = LCMEngine(
        config=LCMConfig(
            fresh_tail_count=2,
            leaf_chunk_tokens=100,
            threshold_full_sweep_enabled=True,
            cold_gap_compaction_seconds=seconds,
            database_path=str(tmp_path / "lcm.db"),
            **overrides,
        ),
        hermes_home=str(tmp_path / "hermes_home"),
    )
    engine.on_session_start("test-session")
    engine.update_model("gpt-test", 1_000_000, provider="openai-codex", api_mode="responses")
    engine.threshold_tokens = THRESHOLD

    def fake_leaf(chunk, focus_topic=None, deadline=None):
        del focus_topic, deadline
        return chunk, count_messages_tokens(chunk), summary, 1, 0

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", fake_leaf)
    return engine


def _messages(rows: int = 20) -> list[dict]:
    messages = [{"role": "system", "content": "system"}]
    for index in range(rows):
        messages.append({
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"cold-gap-{index} " + ("token " * 25),
        })
    return messages


def _provider_call(engine, clock, at: float, prompt_tokens: int = 400_000) -> None:
    clock.now = at
    engine.update_from_response({"prompt_tokens": prompt_tokens, "completion_tokens": 10,
                                 "total_tokens": prompt_tokens + 10})


def _leaves(engine) -> int:
    return engine._last_compress_leaves[1] if engine._last_compress_leaves else 0


def _status(engine) -> dict:
    return json.loads(lcm_tools.lcm_status({}, engine=engine))


@pytest.fixture
def cold(tmp_path, monkeypatch, clock):
    """An engine whose last provider call (400k prompt tokens) is GAP seconds old."""
    engine = _engine(tmp_path, monkeypatch)
    _provider_call(engine, clock, T0)
    clock.now = T0 + GAP
    yield engine
    engine.shutdown()


# -- config ---------------------------------------------------------------------------------------------------------


def test_defaults_are_off_and_300k():
    assert LCMConfig().cold_gap_compaction_seconds == 0
    assert LCMConfig().cold_gap_min_prompt_tokens == 300_000


@pytest.mark.parametrize("raw, expected", [("0", 0), ("300", 300), ("86400", 86400), ("299", 0), ("86401", 0),
                                           ("-1", 0), ("1.5", 0), ("x", 0)])
def test_seconds_env_accepts_0_or_300_to_86400(monkeypatch, raw, expected):
    monkeypatch.setenv("LCM_COLD_GAP_COMPACTION_SECONDS", raw)
    config = LCMConfig.from_env()
    assert config.cold_gap_compaction_seconds == expected
    warned = any("LCM_COLD_GAP_COMPACTION_SECONDS" in w for w in config.config_source_warnings)
    assert warned == (raw not in ("0", "300", "86400"))


@pytest.mark.parametrize("raw, expected", [("1", 1), ("10000000", 10_000_000), ("0", 300_000),
                                           ("10000001", 300_000), ("many", 300_000)])
def test_min_prompt_tokens_env_range(monkeypatch, raw, expected):
    monkeypatch.setenv("LCM_COLD_GAP_MIN_PROMPT_TOKENS", raw)
    assert LCMConfig.from_env().cold_gap_min_prompt_tokens == expected


# -- trigger --------------------------------------------------------------------------------------------------------


def test_setting_0_keeps_todays_preflight_and_compress(tmp_path, monkeypatch, clock):
    messages = _messages()
    results = []
    for seconds in (0, GAP):
        engine = _engine(tmp_path / str(seconds), monkeypatch, seconds=seconds)
        try:
            _provider_call(engine, clock, T0)
            # seconds=GAP with no idle gap is today's path too: the reference for seconds=0.
            clock.now = T0 + (GAP if seconds == 0 else 1)
            due = engine.should_compress_preflight(messages)
            result = engine.compress(messages, current_tokens=count_messages_tokens(messages))
            results.append((due, result, _leaves(engine), engine._compaction_request.cold_gap))
        finally:
            engine.shutdown()
    assert results[0] == results[1]
    assert results[0][0] is False and results[0][3] is False


def test_gap_below_the_setting_does_not_trigger(cold, clock):
    clock.now = T0 + GAP - 1
    assert cold.should_compress_preflight(_messages()) is False
    assert cold._compaction_request.cold_gap is False


def test_cold_gap_compacts_below_the_threshold(cold, caplog):
    messages = _messages()
    with caplog.at_level(logging.INFO):
        assert cold.should_compress_preflight(messages) is True
    assert any(r.getMessage() == (f"LCM cold-gap compaction requested: gap={GAP}s observed_prompt_tokens=400000 "
                                  f"min=300000 threshold={THRESHOLD}") for r in caplog.records)
    result = cold.compress(messages, current_tokens=count_messages_tokens(messages))
    assert _leaves(cold) >= 1
    assert len(result) < len(messages)
    assert count_messages_tokens(result) < count_messages_tokens(messages)
    assert cold.last_compression_status != "sanitized"
    status = _status(cold)
    event = status["cold_gap_compaction"]["last_event"]
    assert status["cold_gap_compaction"]["seconds"] == GAP
    assert status["config"]["cold_gap_compaction_seconds"] == GAP
    assert status["config"]["cold_gap_min_prompt_tokens"] == 300_000
    assert event["gap_seconds"] == GAP and event["observed_prompt_tokens"] == 400_000
    assert event["leaf_passes"] == _leaves(cold)
    assert event["tokens_after"] < event["tokens_before"]
    assert event["stop_reason"] == status["threshold_full_sweep"]["stop_reason"] != ""
    assert event["duration_ms"] >= 0


def test_observed_below_min_does_not_trigger(tmp_path, monkeypatch, clock):
    engine = _engine(tmp_path, monkeypatch)
    try:
        _provider_call(engine, clock, T0, prompt_tokens=299_999)
        clock.now = T0 + GAP
        assert engine.should_compress_preflight(_messages()) is False
    finally:
        engine.shutdown()


def test_no_telemetry_record_does_not_trigger(tmp_path, monkeypatch, clock):
    engine = _engine(tmp_path, monkeypatch)
    try:
        engine.last_prompt_tokens = 400_000
        assert engine._store.read_compaction_telemetry(engine._conversation_id) is None
        assert engine.should_compress_preflight(_messages()) is False
    finally:
        engine.shutdown()


def test_once_per_gap_across_restart_and_rearmed_by_a_new_provider_call(tmp_path, monkeypatch, clock):
    messages = _messages()
    engine = _engine(tmp_path, monkeypatch)
    try:
        _provider_call(engine, clock, T0)
        clock.now = T0 + GAP
        assert engine.should_compress_preflight(messages) is True
        clock.now = T0 + 2 * GAP
        assert engine.should_compress_preflight(messages) is False  # no provider call since
    finally:
        engine.shutdown()
    restarted = _engine(tmp_path, monkeypatch)  # same DB
    try:
        assert restarted.should_compress_preflight(messages) is False
        _provider_call(restarted, clock, T0 + 3 * GAP)
        clock.now = T0 + 4 * GAP
        assert restarted.should_compress_preflight(messages) is True
    finally:
        restarted.shutdown()


@pytest.mark.parametrize("hold", ["no_progress", "sweep_budget", "boundary_cooldown", "route_closed"])
def test_holds_block_the_trigger(cold, monkeypatch, hold):
    if hold == "no_progress":
        cold._start_no_progress_hold("no_progress")
    elif hold == "sweep_budget":
        cold._start_sweep_budget_hold()
    elif hold == "boundary_cooldown":
        cold._last_boundary_skip_time = time.monotonic()
    else:
        monkeypatch.setattr(cold, "_summary_route_available", lambda: False)
    assert cold.should_compress_preflight(_messages()) is False
    assert cold._compaction_request.cold_gap is False
    assert cold._store.read_metadata_json(cold._cold_gap_metadata_key(CONVERSATION)) is None


@pytest.mark.parametrize("kind", ["stateless", "ignored"])
def test_bypassed_sessions_do_not_trigger(cold, kind):
    setattr(cold, f"_session_{kind}", True)
    assert cold.should_compress_preflight(_messages()) is False
    assert cold._compaction_request.cold_gap is False


def test_a_cold_gap_pass_that_stores_nothing_arms_no_hold_and_leaves_the_next_threshold_pass(
        tmp_path, monkeypatch, clock):
    engine = _engine(tmp_path, monkeypatch)
    try:
        _provider_call(engine, clock, T0)
        clock.now = T0 + GAP
        tail_only = _messages(rows=2)  # every row is in the fresh tail: nothing to summarise
        assert engine.should_compress_preflight(tail_only) is True
        engine.compress(tail_only, current_tokens=count_messages_tokens(tail_only))
        assert _leaves(engine) == 0
        assert engine._no_progress_hold is None and engine._no_progress_hold_status() is None
        assert not engine._sweep_budget_hold_active()
        assert not (engine._zero_leaf_streak_status() or {}).get("passes")

        messages = tail_only + _messages()[1:]
        engine.threshold_tokens = count_messages_tokens(messages) - 1
        assert engine.should_compress_preflight(messages) is True
        assert engine._compaction_request.cold_gap is False
        engine.compress(messages, current_tokens=count_messages_tokens(messages))
        assert _leaves(engine) >= 1
    finally:
        engine.shutdown()


def test_at_or_over_the_threshold_runs_todays_threshold_path(tmp_path, monkeypatch, clock):
    messages = _messages()
    outcomes = []
    for seconds in (0, GAP):
        engine = _engine(tmp_path / str(seconds), monkeypatch, seconds=seconds)
        try:
            engine.threshold_tokens = count_messages_tokens(messages) - 1
            _provider_call(engine, clock, T0)
            clock.now = T0 + GAP
            due = engine.should_compress_preflight(messages)
            flagged = engine._compaction_request.cold_gap
            result = engine.compress(messages, current_tokens=count_messages_tokens(messages))
            outcomes.append((due, flagged, result, _leaves(engine),
                             engine._store.read_metadata_json(engine._cold_gap_metadata_key(CONVERSATION))))
        finally:
            engine.shutdown()
    assert outcomes[0] == outcomes[1]
    assert outcomes[1][0] is True and outcomes[1][1] is False and outcomes[1][4] is None
