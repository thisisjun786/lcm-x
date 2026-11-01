"""``LCM_THRESHOLD_FULL_SWEEP_CONDENSATION_RESERVE_PASSES`` / ``lcm.threshold_full_sweep_condensation_reserve_passes``
(T176): passes of the threshold sweep's budget kept for summary-on-summary condensation when the leaf work hits the
pass budget. Env wins over config.yaml; a value outside [0, 16] is ignored with a warning and the next source applies.
``0`` (the default) keeps today's sweep."""

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

FIELD = "threshold_full_sweep_condensation_reserve_passes"
ENV = "LCM_THRESHOLD_FULL_SWEEP_CONDENSATION_RESERVE_PASSES"
YAML_SOURCE = f"config_yaml:lcm.{FIELD}"
LEAF_SUMMARY = "leaf summary words " * 20


@pytest.fixture(autouse=True)
def _pin_lcm_char_counter(monkeypatch):
    """LCM's character estimate for every count, on every import path: a tiktoken encoder whose background load
    finishes between the two sweeps of one test would otherwise change their token counts. The private generation
    keeps the count cache from answering with values counted by an encoder."""
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    monkeypatch.setattr(tokens, "_encoder_generation", -(10**9))


@pytest.fixture
def yaml_lcm(monkeypatch):
    """Pin config.yaml to an ``lcm`` section the test sets (empty by default) and clear the env key."""
    section: dict = {}
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(config, "_load_hermes_config_yaml", lambda *args: {"lcm": section})
    return section


def _warned(c: LCMConfig) -> bool:
    return any(FIELD in w or ENV in w for w in c.config_source_warnings)


# -- config ---------------------------------------------------------------------------------------------------------


def test_unset_keeps_the_default_of_0_with_no_warning(yaml_lcm):
    c = LCMConfig.from_env()
    assert LCMConfig().threshold_full_sweep_condensation_reserve_passes == 0
    assert c.threshold_full_sweep_condensation_reserve_passes == 0
    assert c.config_sources[FIELD] == "default"
    assert not _warned(c)


@pytest.mark.parametrize("raw, value", [("0", 0), ("2", 2), (" 16 ", 16)])
def test_an_env_value_in_range_is_used(yaml_lcm, monkeypatch, raw, value):
    monkeypatch.setenv(ENV, raw)
    c = LCMConfig.from_env()
    assert c.threshold_full_sweep_condensation_reserve_passes == value
    assert c.config_sources[FIELD] == f"env:{ENV}"
    assert not _warned(c)


@pytest.mark.parametrize("raw", [3, "3"])
def test_a_config_yaml_value_in_range_is_used_and_not_reported_as_ignored(yaml_lcm, raw):
    yaml_lcm[FIELD] = raw
    c = LCMConfig.from_env()
    assert c.threshold_full_sweep_condensation_reserve_passes == 3
    assert c.config_sources[FIELD] == YAML_SOURCE
    assert FIELD not in c.ignored_config_yaml_lcm_keys
    assert not _warned(c)


def test_the_env_wins_over_config_yaml(yaml_lcm, monkeypatch):
    yaml_lcm[FIELD] = 3
    monkeypatch.setenv(ENV, "2")
    c = LCMConfig.from_env()
    assert c.threshold_full_sweep_condensation_reserve_passes == 2
    assert c.config_sources[FIELD] == f"env:{ENV}"


@pytest.mark.parametrize("raw", ["17", "-1", "abc", "2.5", ""])
def test_an_env_value_out_of_range_is_rejected_and_0_kept(yaml_lcm, monkeypatch, raw):
    monkeypatch.setenv(ENV, raw)
    c = LCMConfig.from_env()
    assert c.threshold_full_sweep_condensation_reserve_passes == 0
    assert c.config_sources[FIELD] == "default"
    assert _warned(c)


@pytest.mark.parametrize("raw", [17, -1, True, "many", 2.5])
def test_a_config_yaml_value_out_of_range_is_rejected_and_0_kept(yaml_lcm, raw):
    yaml_lcm[FIELD] = raw
    c = LCMConfig.from_env()
    assert c.threshold_full_sweep_condensation_reserve_passes == 0
    assert c.config_sources[FIELD] == "default"
    assert _warned(c)


def test_an_invalid_env_value_falls_back_to_a_valid_config_yaml_value(yaml_lcm, monkeypatch):
    yaml_lcm[FIELD] = 3
    monkeypatch.setenv(ENV, "99")
    c = LCMConfig.from_env()
    assert c.threshold_full_sweep_condensation_reserve_passes == 3
    assert c.config_sources[FIELD] == YAML_SOURCE
    assert _warned(c)


# -- the sweep ------------------------------------------------------------------------------------------------------


class _Clock:
    def __init__(self):
        self._real = time.monotonic
        self.offset = 0.0

    def __call__(self) -> float:
        return self._real() + self.offset


def _sweep(tmp_path, monkeypatch, *, advance_on_leaf=None, clock=None, **overrides):
    """20 raw rows with one-message leaves: only the pass budget stops the leaf work. Each leaf summary is ~60 tokens
    against a 1-token summary-prefix target, so every stored leaf leaves the prefix over target. ``advance_on_leaf``
    moves ``clock`` after that leaf call. Returns (sweep telemetry, lcm_status, summary-prefix tokens after,
    condensation model calls)."""
    settings = {
        "fresh_tail_count": 2,
        "leaf_chunk_tokens": 1,
        "threshold_full_sweep_enabled": True,
        "threshold_full_sweep_max_passes": 6,
        "condensation_fanin": 2,
        "l3_truncate_tokens": 2,  # condensation sources here exceed it, so each condensation is a model call
        "temporal_rollups_enabled": False,
        "foreground_soft_seconds": 60.0,
        "foreground_hard_seconds": 120.0,
        "database_path": str(tmp_path / "lcm.db"),
        **overrides,
    }
    engine = LCMEngine(config=LCMConfig(**settings))
    engine._session_id = "test-session"
    engine.threshold_tokens = 1
    messages = [{"role": "system", "content": "system"}]
    for index in range(20):
        messages.append({
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"budget-{index} " + ("token " * 20),
        })
    leaf_calls, condensation_calls = 0, []

    def fake_leaf(chunk, focus_topic=None, deadline=None):
        nonlocal leaf_calls
        del focus_topic, deadline
        leaf_calls += 1
        if advance_on_leaf is not None and leaf_calls == advance_on_leaf:
            clock.offset += 61.0  # past the 60 s soft target, inside the 120 s hard bound
        return chunk, count_messages_tokens(chunk), LEAF_SUMMARY, 1, 0

    def fake_condensation(**kwargs):
        condensation_calls.append(kwargs.get("source_tokens"))
        return "Condensed.", 1

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", fake_leaf)
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", fake_condensation)
    try:
        engine.compress(messages, current_tokens=count_messages_tokens(messages))
        telemetry = engine.get_status()["threshold_full_sweep"]
        status = json.loads(lcm_tools.lcm_status({}, engine=engine))
        return telemetry, status, engine._summary_frontier_tokens(), condensation_calls
    finally:
        engine.shutdown()


def _stable(telemetry: dict) -> dict:
    return {key: value for key, value in telemetry.items() if key != "duration_ms"}


def test_reserve_0_is_the_sweep_of_today(tmp_path, monkeypatch):
    default, default_status, default_prefix, default_calls = _sweep(tmp_path / "a", monkeypatch)
    zero, zero_status, zero_prefix, zero_calls = _sweep(
        tmp_path / "b", monkeypatch, threshold_full_sweep_condensation_reserve_passes=0)
    assert _stable(zero) == _stable(default)
    assert zero_prefix == default_prefix and zero_calls == default_calls == []
    assert zero["leaf_passes"] == 6 and zero["condensation_passes"] == 0
    assert zero["stop_reason"] == "pass_budget_exhausted"
    assert "reserve_condensation_passes" not in zero
    assert zero_status["config"][FIELD] == default_status["config"][FIELD] == 0


def test_a_capped_sweep_with_reserve_2_condenses_and_ends_with_a_smaller_summary_prefix(tmp_path, monkeypatch):
    base, _status, base_prefix, _calls = _sweep(tmp_path / "r0", monkeypatch)
    telemetry, status, prefix, calls = _sweep(
        tmp_path / "r2", monkeypatch, threshold_full_sweep_condensation_reserve_passes=2)
    assert base["condensation_passes"] == 0
    assert telemetry["leaf_passes"] == 4  # 6 - 2 kept for condensation
    assert telemetry["condensation_passes"] >= 1 and len(calls) == telemetry["condensation_passes"]
    assert telemetry["reserve_condensation_passes"] == telemetry["condensation_passes"]
    assert telemetry["total_passes"] <= 6
    assert telemetry["stop_reason"] == "pass_budget_exhausted"  # the leaf stop is kept
    assert telemetry["summary_prefix_tokens_after"] == prefix < base_prefix
    assert status["config"][FIELD] == 2
    assert status["threshold_full_sweep"]["reserve_condensation_passes"] == telemetry["condensation_passes"]


def test_a_reserve_at_or_over_the_pass_budget_still_leaves_one_leaf_pass(tmp_path, monkeypatch):
    telemetry, _status, _prefix, _calls = _sweep(
        tmp_path, monkeypatch, threshold_full_sweep_max_passes=3, threshold_full_sweep_condensation_reserve_passes=5)
    assert telemetry["leaf_passes"] == 1
    assert telemetry["total_passes"] <= 3
    assert telemetry["stop_reason"] == "pass_budget_exhausted"


def test_no_reserved_condensation_starts_when_the_time_left_is_short(tmp_path, monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    telemetry, _status, _prefix, calls = _sweep(
        tmp_path, monkeypatch, advance_on_leaf=4, clock=clock, threshold_full_sweep_condensation_reserve_passes=2)
    assert telemetry["leaf_passes"] == 4
    assert calls == [] and telemetry["condensation_passes"] == 0
    assert telemetry["reserve_condensation_passes"] == 0
    assert telemetry["reserve_condensation_stop_reason"] == "soft_target_reached"
    assert telemetry["stop_reason"] == "pass_budget_exhausted"
