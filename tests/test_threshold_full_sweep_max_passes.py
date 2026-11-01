"""``LCM_THRESHOLD_FULL_SWEEP_MAX_PASSES`` / ``lcm.threshold_full_sweep_max_passes``: the threshold sweep's
summary-call budget (was a fixed 12). Env wins over config.yaml; a value outside [1, 64] is ignored with a
warning and the next source applies; the sweep and ``lcm_status`` use the value in effect."""

import json

import pytest

import hermes_lcm.config as config
import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens

ENV = "LCM_THRESHOLD_FULL_SWEEP_MAX_PASSES"
YAML_SOURCE = "config_yaml:lcm.threshold_full_sweep_max_passes"


@pytest.fixture
def yaml_lcm(monkeypatch):
    """Pin config.yaml to an ``lcm`` section the test sets (empty by default) and clear the env key."""
    section: dict = {}
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(config, "_load_hermes_config_yaml", lambda *args: {"lcm": section})
    return section


def _warned(c: LCMConfig) -> bool:
    return any("threshold_full_sweep_max_passes" in w or ENV in w for w in c.config_source_warnings)


# -- config ---------------------------------------------------------------------------------------------------------


def test_unset_keeps_the_default_of_12_with_no_warning(yaml_lcm):
    c = LCMConfig.from_env()
    assert LCMConfig().threshold_full_sweep_max_passes == 12
    assert c.threshold_full_sweep_max_passes == 12
    assert c.config_sources["threshold_full_sweep_max_passes"] == "default"
    assert not _warned(c)


@pytest.mark.parametrize("raw, value", [("1", 1), ("24", 24), (" 64 ", 64)])
def test_an_env_value_in_range_is_used(yaml_lcm, monkeypatch, raw, value):
    monkeypatch.setenv(ENV, raw)
    c = LCMConfig.from_env()
    assert c.threshold_full_sweep_max_passes == value
    assert c.config_sources["threshold_full_sweep_max_passes"] == f"env:{ENV}"
    assert not _warned(c)


@pytest.mark.parametrize("raw", [24, "24"])
def test_a_config_yaml_value_in_range_is_used_and_not_reported_as_ignored(yaml_lcm, raw):
    yaml_lcm["threshold_full_sweep_max_passes"] = raw
    c = LCMConfig.from_env()
    assert c.threshold_full_sweep_max_passes == 24
    assert c.config_sources["threshold_full_sweep_max_passes"] == YAML_SOURCE
    assert "threshold_full_sweep_max_passes" not in c.ignored_config_yaml_lcm_keys
    assert not _warned(c)


def test_the_env_wins_over_config_yaml(yaml_lcm, monkeypatch):
    yaml_lcm["threshold_full_sweep_max_passes"] = 24
    monkeypatch.setenv(ENV, "20")
    c = LCMConfig.from_env()
    assert c.threshold_full_sweep_max_passes == 20
    assert c.config_sources["threshold_full_sweep_max_passes"] == f"env:{ENV}"


@pytest.mark.parametrize("raw", ["0", "65", "-1", "abc", "2.5", ""])
def test_an_env_value_out_of_range_is_rejected_and_12_kept(yaml_lcm, monkeypatch, raw):
    monkeypatch.setenv(ENV, raw)
    c = LCMConfig.from_env()
    assert c.threshold_full_sweep_max_passes == 12
    assert c.config_sources["threshold_full_sweep_max_passes"] == "default"
    assert _warned(c)


@pytest.mark.parametrize("raw", [0, 65, True, "many", 2.5])
def test_a_config_yaml_value_out_of_range_is_rejected_and_12_kept(yaml_lcm, raw):
    yaml_lcm["threshold_full_sweep_max_passes"] = raw
    c = LCMConfig.from_env()
    assert c.threshold_full_sweep_max_passes == 12
    assert c.config_sources["threshold_full_sweep_max_passes"] == "default"
    assert _warned(c)


def test_an_invalid_env_value_falls_back_to_a_valid_config_yaml_value(yaml_lcm, monkeypatch):
    yaml_lcm["threshold_full_sweep_max_passes"] = 24
    monkeypatch.setenv(ENV, "100")
    c = LCMConfig.from_env()
    assert c.threshold_full_sweep_max_passes == 24
    assert c.config_sources["threshold_full_sweep_max_passes"] == YAML_SOURCE
    assert _warned(c)


# -- the sweep and lcm_status use the value -------------------------------------------------------------------------


def _sweep(tmp_path, monkeypatch, **overrides) -> tuple[dict, dict]:
    """20 raw rows with one-token leaves (the shared-pass-budget fixture): only the pass budget stops the sweep."""
    engine = LCMEngine(config=LCMConfig(
        fresh_tail_count=2,
        leaf_chunk_tokens=1,
        threshold_full_sweep_enabled=True,
        database_path=str(tmp_path / "lcm.db"),
        **overrides,
    ))
    engine._session_id = "test-session"
    engine.threshold_tokens = 1
    messages = [{"role": "system", "content": "system"}]
    for index in range(20):
        messages.append({
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"budget-{index} " + ("token " * 20),
        })

    def fake_leaf(chunk, focus_topic=None, deadline=None):
        del focus_topic, deadline
        return chunk, count_messages_tokens(chunk), "bounded", 1, 0

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", fake_leaf)
    try:
        engine.compress(messages, current_tokens=count_messages_tokens(messages))
        return engine.get_status()["threshold_full_sweep"], json.loads(lcm_tools.lcm_status({}, engine=engine))
    finally:
        engine.shutdown()


@pytest.mark.parametrize("passes", [3, 16])
def test_the_sweep_stops_at_the_configured_pass_budget_and_status_reports_it(tmp_path, monkeypatch, passes):
    telemetry, status = _sweep(tmp_path, monkeypatch, threshold_full_sweep_max_passes=passes)
    assert telemetry["leaf_passes"] == passes
    assert telemetry["total_passes"] == passes
    assert telemetry["stop_reason"] == "pass_budget_exhausted"
    assert status["config"]["threshold_full_sweep_max_passes"] == passes


def test_the_default_sweep_still_stops_at_12_and_status_reports_12(tmp_path, monkeypatch):
    telemetry, status = _sweep(tmp_path, monkeypatch)
    assert telemetry["total_passes"] == 12
    assert telemetry["stop_reason"] == "pass_budget_exhausted"
    assert status["config"]["threshold_full_sweep_max_passes"] == 12
