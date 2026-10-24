"""fork (#1162 family, 10-25 hub move): a cold bind whose commit proof no longer binds (the host moved, so
``hermes_home`` changed) handed LCM its own emitted summary head as the first host row. Ingest stored that head
as a raw user row above the frontier; the in-order store mapper then tied the host's head to that copy, every
later host row went unmapped, and each compaction stopped on "leaf would end before an unresolved retained
occurrence (... kind=shown)" (sion 20261024_153651_55951e, default 20261015_004705_713a68).

Covered here: the guard (a host row carrying an engine uid LCM minted for this session root, headed by a verified
summary, is not stored again; a user's own summary-shaped row still is) and the lossless repair for a store an
earlier cold bind already wedged (``fork_summary_quarantine``)."""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest

import hermes_lcm.engine as lcm_engine_module
from hermes_lcm import host_uid_emit as emit
from hermes_lcm.engine import LCMEngine
from tests.test_compression_boundary import _config, _stub_summarizer, _turn
from tests.test_host_uid_shadow import _state_db

BASE = 1_792_689_000.0
_spec = importlib.util.spec_from_file_location(
    "fork_summary_quarantine", Path(__file__).resolve().parent.parent / "fork_summary_quarantine.py")
fsq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fsq)


@pytest.fixture(autouse=True)
def carry_mode(monkeypatch):
    monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    monkeypatch.setattr(emit, "_host_uid_capability", True)


def _home(tmp_path, name):
    home = tmp_path / name
    home.mkdir(exist_ok=True)
    _state_db(home, [("S0", None, None, "cli")])
    return str(home)


def _stamped(messages, start):
    return [dict(m, timestamp=BASE + start + i, message_uid=f"u{start + i}") for i, m in enumerate(messages)]


def _compacted(tmp_path, monkeypatch):
    """Compacted on the old host: the summary head LCM returns carries an engine uid it recorded."""
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
    engine = LCMEngine(config=_config(tmp_path, fresh_tail_count=2), hermes_home=_home(tmp_path, "old"))
    engine.on_session_start("S0", platform="cli", context_length=200_000)
    host = []
    for turn in range(1, 13):
        host.extend(_stamped(_turn(turn), 2 * turn))
        engine.ingest(host)
    compressed = engine.compress(list(host), force=True)
    assert engine._last_compression_status == "compacted"
    head = next(m for m in compressed if m.get("role") == "user"
                and engine._verified_lcm_summary_prefix_end(str(m.get("content") or "")) is not None)
    assert head.get("message_uid")
    return engine, compressed, head


def _moved_host(engine, tmp_path):
    """The same lcm.db opened from a new hermes_home (the hub move): the old commit proof no longer binds."""
    engine.shutdown()
    moved = LCMEngine(config=_config(tmp_path, fresh_tail_count=2), hermes_home=_home(tmp_path, "moved"))
    moved.on_session_start("S0", platform="cli", context_length=200_000)
    return moved


def _persisted_view(compressed, stamp):
    """What the moved host replays: system prompt, then state.db's rows (the head keeps its engine uid)."""
    view = [{"role": "system", "content": "You are concise."}]
    for message in compressed:
        row = dict(message)
        if row.get("role") == "user" and "timestamp" not in row:
            row["timestamp"] = stamp
        view.append(row)
    return view


def _count_content(engine, content):
    return engine._store._conn.execute(
        "SELECT count(*) FROM messages WHERE session_id = 'S0' AND content = ?", (content,)).fetchone()[0]


def _frontier(engine):
    return int(engine._lifecycle.get_by_conversation(engine._conversation_id).current_frontier_store_id or 0)


def test_cold_bind_does_not_store_lcm_summary_head(tmp_path, monkeypatch):
    engine, compressed, head = _compacted(tmp_path, monkeypatch)
    assert _count_content(engine, head["content"]) == 0
    engine = _moved_host(engine, tmp_path)
    try:
        view = _persisted_view(compressed, BASE + 100.5)
        view.append({"role": "user", "content": "after the move", "timestamp": BASE + 101, "message_uid": "new1"})
        engine.ingest(view)
        assert _count_content(engine, head["content"]) == 0  # LCM's own head is not stored as content
        assert _count_content(engine, "after the move") == 1
    finally:
        engine.shutdown()


def test_guard_off_reproduces_the_stored_head(tmp_path, monkeypatch):
    engine, compressed, head = _compacted(tmp_path, monkeypatch)
    monkeypatch.setattr(lcm_engine_module, "_FORK_NEVER_STORE_OWN_SUMMARY", False)
    engine = _moved_host(engine, tmp_path)
    try:
        engine.ingest(_persisted_view(compressed, BASE + 100.5))
        (store_id,) = engine._store._conn.execute(
            "SELECT store_id FROM messages WHERE session_id = 'S0' AND content = ?", (head["content"],)).fetchone()
        assert int(store_id) > _frontier(engine)  # the wedge's precondition, as stored on 10-25
    finally:
        engine.shutdown()


def test_summary_shaped_row_without_engine_uid_is_still_content(tmp_path, monkeypatch):
    engine, _compressed, head = _compacted(tmp_path, monkeypatch)
    engine = _moved_host(engine, tmp_path)
    try:
        pasted = {"role": "user", "content": head["content"], "timestamp": BASE + 200, "message_uid": "user-paste"}
        assert not engine._fork_is_own_emission(pasted)  # the user's own row: no engine uid
        assert not engine._fork_is_own_emission(dict(pasted, message_uid=None))
        assert engine._fork_is_own_emission(dict(pasted, message_uid=head["message_uid"]))
    finally:
        engine.shutdown()


def test_own_carrier_is_dropped_only_when_its_user_row_is_stored(tmp_path, monkeypatch):
    engine, compressed, head = _compacted(tmp_path, monkeypatch)
    engine = _moved_host(engine, tmp_path)
    try:
        stored_user = next(m for m in compressed if m.get("role") == "user" and m is not head)["content"]
        carrier = {"role": "user", "content": head["content"] + "\n\n" + stored_user, "timestamp": BASE + 100.7,
                   "message_uid": head["message_uid"]}
        assert engine._fork_is_own_emission(carrier)
        fresh = dict(carrier, content=head["content"] + "\n\nbrand new words")
        assert not engine._fork_is_own_emission(fresh)  # a new user row inside the carrier is content
    finally:
        engine.shutdown()


def _wedged_store(tmp_path, monkeypatch):
    """A store the old path wedged: the moved host stored the head above the frontier, then kept working."""
    engine, compressed, head = _compacted(tmp_path, monkeypatch)
    monkeypatch.setattr(lcm_engine_module, "_FORK_NEVER_STORE_OWN_SUMMARY", False)
    engine = _moved_host(engine, tmp_path)
    view = _persisted_view(compressed, BASE + 100.5)
    engine.ingest(view)
    for turn in range(20, 32):
        view.extend(_stamped(_turn(turn), 2 * turn))
        engine.ingest(view)
    return engine, view, head


def test_quarantine_repairs_a_wedged_store_losslessly(tmp_path, monkeypatch):
    engine, _view, head = _wedged_store(tmp_path, monkeypatch)
    db = engine._store.db_path
    try:
        (stored,) = engine._store._conn.execute(
            "SELECT store_id FROM messages WHERE session_id = 'S0' AND content = ?", (head["content"],)).fetchone()
        stored = int(stored)
        total_before = engine._store._conn.execute("SELECT count(*) FROM messages").fetchone()[0]
    finally:
        engine.shutdown()
    conn = sqlite3.connect(db, isolation_level=None)
    planned = fsq.plan(conn, "S0")
    assert [sid for sid, _n, _c in planned["rows"]] == [stored]
    fsq.apply(conn, "S0", reason="test")
    # lossless: same row count, same bytes, only the owner moved; recorded for restore
    assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == total_before
    row = tuple(conn.execute("SELECT session_id, content FROM messages WHERE store_id = ?", (stored,)).fetchone())
    assert row == ("S0" + fsq.QUARANTINE_SUFFIX, head["content"])
    assert conn.execute(f"SELECT original_session_id FROM {fsq.TABLE} WHERE store_id = ?", (stored,)).fetchone()[0] == "S0"
    assert fsq.plan(conn, "S0")["rows"] == []  # idempotent
    assert fsq.restore(conn, [stored]) == 1
    assert conn.execute("SELECT session_id FROM messages WHERE store_id = ?", (stored,)).fetchone()[0] == "S0"
    conn.close()


def test_pure_summary_check_is_strict(tmp_path, monkeypatch):
    engine, _compressed, head = _compacted(tmp_path, monkeypatch)
    db = engine._store.db_path
    engine.shutdown()
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    block = head["content"]
    assert fsq.is_pure_own_summary(conn, "S0", block)
    assert fsq.is_pure_own_summary(conn, "S0", block + "\n\nuser words") is None  # a carrier is not pure
    assert fsq.is_pure_own_summary(conn, "OTHER", block) is None  # another session's nodes are content
    assert fsq.is_pure_own_summary(conn, "S0", block.replace("\n", " ", 1)) is None  # edited header
    conn.close()
