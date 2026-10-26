"""The tiktoken BPE file must be cached somewhere durable.

tiktoken's default cache is under $TMPDIR, which macOS purges; the next
process then re-downloads on its first count. The manifest declaration that
keeps tiktoken installed is pinned in test_issue_631_not_a_pm_member.py.
"""

from pathlib import Path

import pytest

from hermes_lcm import tokens as tokens_mod

@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in ("TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    return tmp_path


def test_cache_dir_defaults_under_hermes_home(clean_env):
    target = tokens_mod._tiktoken_cache_dir()
    assert target == str(clean_env / "home" / "cache" / "tiktoken")
    assert Path(target).is_dir()


@pytest.mark.parametrize("key", ["TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR"])
@pytest.mark.parametrize("value", ["/somewhere/else", ""])
def test_operator_choice_wins(clean_env, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    assert tokens_mod._tiktoken_cache_dir() is None
    assert not (clean_env / "home" / "cache").exists()


def test_unwritable_home_falls_back_to_tiktoken_default(clean_env, monkeypatch):
    blocker = clean_env / "home"
    blocker.write_text("not a directory")
    assert tokens_mod._tiktoken_cache_dir() is None


def test_load_encoder_points_tiktoken_at_the_durable_cache(clean_env, monkeypatch):
    import os
    import sys
    import types

    seen = {}
    fake = types.ModuleType("tiktoken")
    fake.get_encoding = lambda name: seen.setdefault("call", (name, os.environ.get("TIKTOKEN_CACHE_DIR")))
    monkeypatch.setitem(sys.modules, "tiktoken", fake)
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    tokens_mod._load_encoder()
    assert seen["call"] == ("cl100k_base", str(clean_env / "home" / "cache" / "tiktoken"))
