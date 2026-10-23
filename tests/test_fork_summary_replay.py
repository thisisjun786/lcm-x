"""fork: a CLI resume hands LCM its own summary row (``[Recent Summary ...]``, the boundary it rendered
earlier) with the same bytes and host stamp every time. The identity anchor gives a pure scaffold row no
identity, so before this fix each resume stored that row again (27 copies of one 57k-char summary in the
sion session of 10-23). Same stamp + same bytes already stored in this lineage = a replay, not content."""

from __future__ import annotations

import hermes_lcm.engine as lcm_engine_module
from hermes_lcm.engine import LCMEngine
from tests.test_compression_boundary import _config, _stub_summarizer, _turn
from tests.test_issue_488_emission_proof import _summary_block

BASE = 1_792_689_000.0


def _stamped(messages, start):
    return [dict(m, timestamp=BASE + start + i) for i, m in enumerate(messages)]


def _compacted(tmp_path, monkeypatch):
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
    engine = LCMEngine(config=_config(tmp_path, fresh_tail_count=2), hermes_home=str(tmp_path / "home"))
    engine.on_session_start("S0", platform="cli", context_length=200_000)
    host = []
    for turn in range(1, 13):
        host.extend(_stamped(_turn(turn), 2 * turn))
        engine.ingest(host)
    compressed = engine.compress(list(host), force=True)
    assert engine._last_compression_status == "compacted"
    return engine, compressed, _summary_block(engine, compressed)


def _resume(engine, tmp_path):
    engine.shutdown()
    resumed = LCMEngine(config=_config(tmp_path, fresh_tail_count=2), hermes_home=str(tmp_path / "home"))
    resumed.on_session_start("S0", platform="cli", context_length=200_000)
    return resumed


def _count(engine, sql, *args):
    return engine._store._conn.execute(sql, args).fetchone()[0]


def _persisted_view(compressed, block, stamp):
    """What a CLI resume hands the engine: the system prompt, then state.db's rows, where the boundary's
    summary row is a stamped user row (the system row puts the summary past the anchor's head skip)."""
    view = [{"role": "system", "content": "You are concise."}]
    for message in compressed:
        if message.get("role") == "user" and str(message.get("content") or "").startswith(block):
            view.append(dict(message, content=block, timestamp=stamp))
        else:
            view.append(dict(message))
    return view


def test_resumed_summary_row_with_same_stamp_is_stored_once(tmp_path, monkeypatch):
    engine, compressed, block = _compacted(tmp_path, monkeypatch)
    stamp = BASE + 100.5
    view = _persisted_view(compressed, block, stamp)
    try:
        engine = _resume(engine, tmp_path)
        engine.ingest([dict(m) for m in view])  # the first resume stores the summary row once (as today)
        first = _count(engine, "SELECT count(*) FROM messages WHERE role = 'user' AND content = ?", block)
        assert first == 1
        for attempt in range(3):  # every later resume replays the same row from cursor 0
            view = view + [{"role": "user", "content": f"resume prompt {attempt}", "timestamp": stamp + 10 + attempt}]
            engine = _resume(engine, tmp_path)
            engine.ingest([dict(m) for m in view])
        assert _count(engine, "SELECT count(*) FROM messages WHERE role = 'user' AND content = ?", block) == first
        assert _count(engine, "SELECT count(*) FROM messages WHERE content LIKE 'resume prompt %'") == 3
    finally:
        engine.shutdown()


def test_summary_row_check_needs_the_same_stamp_and_bytes(tmp_path, monkeypatch):
    engine, compressed, block = _compacted(tmp_path, monkeypatch)
    stamp = BASE + 100.5
    try:
        engine = _resume(engine, tmp_path)
        engine.ingest(_persisted_view(compressed, block, stamp))
        assert _count(engine, "SELECT count(*) FROM messages WHERE role = 'user' AND content = ?", block) == 1
        check = engine._fork_scaffold_replayed_indexes
        assert check([{"role": "user", "content": block, "timestamp": stamp}], 0) == {0}
        assert check([{"role": "user", "content": block, "timestamp": stamp + 5}], 0) == set()  # other stamp
        assert check([{"role": "user", "content": block}], 0) == set()  # no stamp
        assert check([{"role": "user", "content": block + "\n\nnew words", "timestamp": stamp}], 0) == set()
        assert check([{"role": "user", "content": "plain text", "timestamp": stamp}], 0) == set()  # not a summary
    finally:
        engine.shutdown()
