"""``LCM_SWEEP_TARGET_PROMPT_TOKENS`` / ``lcm.sweep_target_prompt_tokens`` (G60): the threshold sweep works toward a
host-counted prompt size. The LCM count is scaled by the host/LCM ratio observed at sweep entry; without a usable
ratio there is no target decision. ``0`` (the default) keeps today's sweep."""

import json
import time

import pytest

import hermes_lcm.config as config
import hermes_lcm.engine as lcm_engine
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens

FIELD = "sweep_target_prompt_tokens"
ENV = "LCM_SWEEP_TARGET_PROMPT_TOKENS"
TARGET_KEYS = {"target_prompt_tokens", "prompt_token_ratio", "projected_prompt_tokens_after"}


@pytest.fixture(autouse=True)
def _pin_lcm_char_counter(monkeypatch):
    """One counter for every count in a test (see test_threshold_full_sweep_condensation_reserve)."""
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    monkeypatch.setattr(tokens, "_encoder_generation", -(10**9))


@pytest.fixture
def yaml_lcm(monkeypatch):
    section: dict = {}
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(config, "_load_hermes_config_yaml", lambda *args: {"lcm": section})
    return section


# -- config ---------------------------------------------------------------------------------------------------------


def test_unset_is_0(yaml_lcm):
    c = LCMConfig.from_env()
    assert LCMConfig().sweep_target_prompt_tokens == c.sweep_target_prompt_tokens == 0
    assert c.config_sources[FIELD] == "default"


def test_env_wins_over_config_yaml(yaml_lcm, monkeypatch):
    yaml_lcm[FIELD] = 200000
    monkeypatch.setenv(ENV, "150000")
    c = LCMConfig.from_env()
    assert c.sweep_target_prompt_tokens == 150000
    assert c.config_sources[FIELD] == f"env:{ENV}"


@pytest.mark.parametrize("raw", ["-1", "4000001", "abc", "1.5e5"])
def test_an_invalid_env_value_is_ignored_with_a_warning(yaml_lcm, monkeypatch, raw):
    yaml_lcm[FIELD] = 120000
    monkeypatch.setenv(ENV, raw)
    c = LCMConfig.from_env()
    assert c.sweep_target_prompt_tokens == 120000
    assert c.config_sources[FIELD] == f"config_yaml:lcm.{FIELD}"
    assert any(ENV in w for w in c.config_source_warnings)


# -- the sweep ------------------------------------------------------------------------------------------------------


class _Clock:
    def __init__(self):
        self._real = time.monotonic
        self.offset = 0.0

    def __call__(self) -> float:
        return self._real() + self.offset


def _messages():
    messages = [{"role": "system", "content": "system"}]
    for index in range(20):
        messages.append({
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"turn-{index} " + ("word " * 200),
        })
    return messages


def _sweep(tmp_path, monkeypatch, *, ratio=1.6, host_tokens=None, advance_on_leaf=None, clock=None, **overrides):
    """20 raw rows of ~250 LCM tokens, one-message leaves with a short summary, so every leaf frees ~245 tokens.
    The host count is ``ratio`` x the LCM count unless ``host_tokens`` is given. Returns (sweep telemetry,
    lcm_status, LCM count at entry, condensation model calls)."""
    settings = {
        "fresh_tail_count": 2,
        "leaf_chunk_tokens": 1,
        "threshold_full_sweep_enabled": True,
        "threshold_full_sweep_max_passes": 6,
        "condensation_fanin": 2,
        "l3_truncate_tokens": 0,
        "temporal_rollups_enabled": False,
        "foreground_soft_seconds": 60.0,
        "foreground_hard_seconds": 120.0,
        "database_path": str(tmp_path / "lcm.db"),
        **overrides,
    }
    engine = LCMEngine(config=LCMConfig(**settings))
    engine._session_id = "test-session"
    engine.threshold_tokens = 1
    messages = _messages()
    lcm_tokens = count_messages_tokens(messages)
    leaf_calls, condensation_calls = 0, []

    def fake_leaf(chunk, focus_topic=None, deadline=None):
        nonlocal leaf_calls
        del focus_topic, deadline
        leaf_calls += 1
        if advance_on_leaf is not None and leaf_calls == advance_on_leaf:
            clock.offset += 61.0  # past the 60 s soft target, inside the 120 s hard bound
        return chunk, count_messages_tokens(chunk), "leaf summary of the turn", 1, 0

    def fake_condensation(**kwargs):
        condensation_calls.append(kwargs.get("source_tokens"))
        return "Condensed.", 1

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", fake_leaf)
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", fake_condensation)
    try:
        observed = host_tokens if host_tokens is not None else int(lcm_tokens * ratio)
        engine.compress(messages, current_tokens=observed)
        telemetry = engine.get_status()["threshold_full_sweep"]
        status = json.loads(lcm_tools.lcm_status({}, engine=engine))
        return telemetry, status, lcm_tokens, condensation_calls
    finally:
        engine.shutdown()


def _stable(telemetry: dict) -> dict:
    return {key: value for key, value in telemetry.items() if key != "duration_ms"}


def test_target_0_is_the_sweep_of_today(tmp_path, monkeypatch):
    default, default_status, _lcm, default_calls = _sweep(tmp_path / "a", monkeypatch)
    zero, zero_status, _lcm, zero_calls = _sweep(tmp_path / "b", monkeypatch, sweep_target_prompt_tokens=0)
    assert _stable(zero) == _stable(default)
    assert zero_calls == default_calls
    assert not TARGET_KEYS & set(zero)
    assert zero["leaf_passes"] == 6 and zero["stop_reason"] == "pass_budget_exhausted"
    assert zero_status["config"][FIELD] == default_status["config"][FIELD] == 0


def test_leaves_stop_once_the_projected_request_fits_the_target(tmp_path, monkeypatch):
    lcm_tokens = count_messages_tokens(_messages())
    target = int(lcm_tokens * 1.6) - 600  # about two leaves of host tokens
    telemetry, status, _lcm, _calls = _sweep(tmp_path, monkeypatch, sweep_target_prompt_tokens=target)
    assert telemetry["stop_reason"] == "target_reached"
    assert telemetry["status"] == "completed"
    assert 1 <= telemetry["leaf_passes"] < 6
    assert telemetry["prompt_token_ratio"] == pytest.approx(1.6, abs=0.001)
    assert telemetry["target_prompt_tokens"] == target
    assert telemetry["projected_prompt_tokens_after"] == pytest.approx(
        telemetry["tokens_after"] * telemetry["prompt_token_ratio"], rel=1e-3)
    assert telemetry["projected_prompt_tokens_after"] <= target
    assert status["config"][FIELD] == target
    assert status["threshold_full_sweep"]["projected_prompt_tokens_after"] == telemetry["projected_prompt_tokens_after"]


def test_the_ratio_scales_the_decision(tmp_path, monkeypatch):
    """The same target is reached after fewer leaves when host tokens per LCM token are fewer."""
    lcm_tokens = count_messages_tokens(_messages())
    target = int(lcm_tokens * 1.6) - 600
    high, *_ = _sweep(tmp_path / "high", monkeypatch, ratio=1.6, sweep_target_prompt_tokens=target)
    low, *_ = _sweep(tmp_path / "low", monkeypatch, ratio=1.55, sweep_target_prompt_tokens=target)
    assert low["leaf_passes"] < high["leaf_passes"]


def test_a_target_out_of_reach_stops_on_the_pass_budget(tmp_path, monkeypatch):
    telemetry, _status, _lcm, _calls = _sweep(tmp_path, monkeypatch, sweep_target_prompt_tokens=1)
    assert telemetry["leaf_passes"] == 6
    assert telemetry["stop_reason"] == "pass_budget_exhausted"
    assert telemetry["status"] == "partial"
    assert telemetry["projected_prompt_tokens_after"] > 1


def test_a_target_out_of_reach_stops_on_time(tmp_path, monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    telemetry, _status, _lcm, _calls = _sweep(
        tmp_path, monkeypatch, advance_on_leaf=2, clock=clock, sweep_target_prompt_tokens=1)
    assert telemetry["leaf_passes"] == 2
    assert telemetry["stop_reason"] in {"soft_target_reached", "time_budget_exhausted"}


@pytest.mark.parametrize("host_tokens", [0, "tenfold"])
def test_no_usable_ratio_means_no_target_decision(tmp_path, monkeypatch, host_tokens):
    """No host count, or one ten times the LCM count (a stale observation), leaves only the pass and time bounds."""
    lcm_tokens = count_messages_tokens(_messages())
    observed = lcm_tokens * 10 if host_tokens == "tenfold" else 0
    engine_threshold_kwargs = {"sweep_target_prompt_tokens": lcm_tokens}  # would be reached at once with a ratio
    telemetry, _status, _lcm, _calls = _sweep(tmp_path, monkeypatch, host_tokens=observed, **engine_threshold_kwargs)
    assert telemetry["prompt_token_ratio"] is None and telemetry["projected_prompt_tokens_after"] is None
    assert telemetry["leaf_passes"] == 6
    assert telemetry["stop_reason"] == "pass_budget_exhausted"


def test_a_target_condenses_below_the_frontier_target_within_the_reserve(tmp_path, monkeypatch):
    """With a frontier target the leaves never reach, T176's reserve makes no call; a request still over the prompt
    target condenses anyway, and the pass budget still bounds the sweep."""
    base, *_rest, base_calls = _sweep(
        tmp_path / "base", monkeypatch, summary_prefix_target_tokens=100000,
        threshold_full_sweep_condensation_reserve_passes=2)
    telemetry, _status, _lcm, calls = _sweep(
        tmp_path / "target", monkeypatch, summary_prefix_target_tokens=100000,
        threshold_full_sweep_condensation_reserve_passes=2, sweep_target_prompt_tokens=1)
    assert base_calls == [] and base["condensation_passes"] == 0
    assert telemetry["leaf_passes"] == 4
    assert 1 <= len(calls) == telemetry["condensation_passes"] <= 2
    assert telemetry["total_passes"] <= 6
    assert telemetry["summary_prefix_tokens_after"] < base["summary_prefix_tokens_after"]
    assert telemetry["stop_reason"] == "pass_budget_exhausted"


def test_nothing_left_to_summarise_or_condense_is_target_unreachable(tmp_path, monkeypatch):
    telemetry, _status, _lcm, calls = _sweep(
        tmp_path, monkeypatch, threshold_full_sweep_max_passes=64, sweep_target_prompt_tokens=1)
    assert telemetry["leaf_passes"] == 18  # the raw prefix outside the 2-row fresh tail
    assert len(calls) == telemetry["condensation_passes"] >= 1
    assert telemetry["total_passes"] < 64
    assert telemetry["stop_reason"] == "target_unreachable"
    assert telemetry["status"] == "partial"
