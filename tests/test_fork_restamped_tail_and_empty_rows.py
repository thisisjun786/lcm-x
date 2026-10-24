"""fork (10-25, default telegram session 20261024_165049_398f126f): compaction stopped with
``raw_prefix_drained`` while ~160k raw tokens stayed above the frontier. Three causes, each pinned here:

1. Hermes' compression re-persists the kept tail with FRESH timestamps. #436 matches by stamp + payload, and the
   fork's uid-replay check only took unstamped rows, so the first ingest after every compression stored the whole
   tail again (634 rows at 07:09). After the hub move the new-home lineage key also had no binding for the old
   rows. Now a row whose message_uid is bound (any home tag of this session root) to a stored row with the same
   bytes is a replay even when its stamp changed; a row whose stored copy has its current stamp stays #436's.
2. Rows with an empty identity (a content-less assistant turn) matched ANY empty view occurrence, so a later empty
   reply "showed" an older one and the leaf stopped there. A view occurrence whose uid is bound now pairs only
   with its own row.
3. A composite carrier (one host row holding stored user turns a and c, with an empty assistant b between) sat
   behind host rows mapped to later store rows, so its constituents were never emitted in store order and the
   contiguity check cut every leaf to nothing. Its constituents are now read from their stored bytes at their
   own store positions, as single out-of-order claims already were."""

from __future__ import annotations

import sqlite3

import pytest

from hermes_lcm import host_uid_emit as emit
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
import hermes_lcm.engine as lcm_engine_module

PAD = " alpha beta gamma delta" * 4


@pytest.fixture(autouse=True)
def carry_mode(monkeypatch):
    monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    monkeypatch.setattr(emit, "_host_uid_capability", True)
    monkeypatch.setattr(lcm_engine_module, "_FORK_UID_REPLAY_STAMPED", True)  # opt-in in this commit


def _home(tmp_path, name):
    home = tmp_path / name
    home.mkdir(exist_ok=True)
    conn = sqlite3.connect(home / "state.db")
    conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, "
                 "end_reason TEXT, source TEXT, model_config TEXT)")
    conn.execute("INSERT OR REPLACE INTO sessions VALUES ('S', NULL, NULL, NULL, NULL)")
    conn.commit()
    conn.close()
    return str(home)


def _engine(tmp_path, home="h1") -> LCMEngine:
    engine = LCMEngine(config=LCMConfig(fresh_tail_count=2, leaf_chunk_tokens=1,
                                        database_path=str(tmp_path / "lcm.db")), hermes_home=_home(tmp_path, home))
    engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
    return engine


def _history():
    call = {"id": "call_1", "type": "function", "function": {"name": "terminal", "arguments": "{\"command\":\"ls\"}"}}
    return [
        {"role": "user", "content": "check the folder" + PAD, "timestamp": 10.0, "message_uid": "u-1"},
        {"role": "assistant", "content": "", "tool_calls": [call], "timestamp": 11.0, "message_uid": "u-2"},
        {"role": "tool", "content": "{\"output\": \"a.txt b.txt\"}" + PAD, "tool_call_id": "call_1",
         "timestamp": 11.5, "message_uid": "u-3"},
        {"role": "assistant", "content": "two files" + PAD, "timestamp": 12.0, "message_uid": "u-4"},
    ]


def _restamped(history, stamp):
    """Hermes' compression boundary: the kept tail re-persisted with one fresh timestamp per row."""
    return [dict(m, timestamp=stamp + i * 1e-6) for i, m in enumerate(history)]


def _count(engine, content) -> int:
    return engine._store._conn.execute(
        "SELECT count(*) FROM messages WHERE session_id = 'S' AND content = ?", (content,)).fetchone()[0]


def _lead():
    return {"role": "user", "content": "[Recent Summary] carried" + PAD, "timestamp": 9.0, "message_uid": "u-0"}


def test_restamped_tail_after_compression_is_not_stored_again(tmp_path):
    first = _engine(tmp_path)
    try:
        first.ingest(_history())
    finally:
        first.shutdown()
    again = _engine(tmp_path)
    try:
        again.ingest([_lead(), *_restamped(_history(), 500.0)])
        assert _count(again, "two files" + PAD) == 1
        assert _count(again, "{\"output\": \"a.txt b.txt\"}" + PAD) == 1
    finally:
        again.shutdown()


def test_restamped_tail_after_a_host_move_is_not_stored_again(tmp_path):
    first = _engine(tmp_path, "old-home")
    try:
        first.ingest(_history())
    finally:
        first.shutdown()
    moved = _engine(tmp_path, "new-home")  # same lcm.db, new hermes_home: a new lineage home tag
    try:
        moved.ingest([_lead(), *_restamped(_history(), 500.0)])
        assert _count(moved, "two files" + PAD) == 1
    finally:
        moved.shutdown()


def test_restamped_row_with_new_bytes_is_still_stored(tmp_path):
    first = _engine(tmp_path)
    try:
        first.ingest(_history())
    finally:
        first.shutdown()
    again = _engine(tmp_path)
    try:
        edited = _restamped(_history(), 500.0)
        edited[3] = {**edited[3], "content": "three files" + PAD}  # same uid, new bytes: an edit, kept
        again.ingest([_lead(), *edited])
        assert _count(again, "three files" + PAD) == 1
        assert _count(again, "two files" + PAD) == 1
    finally:
        again.shutdown()


def test_stamped_check_is_off_by_default_switch(tmp_path, monkeypatch):
    first = _engine(tmp_path)
    try:
        first.ingest(_history())
    finally:
        first.shutdown()
    monkeypatch.setattr(lcm_engine_module, "_FORK_UID_REPLAY_STAMPED", False)
    again = _engine(tmp_path)
    try:
        assert again._fork_uid_replayed_indexes(_restamped(_history(), 500.0), 0) == set()
    finally:
        again.shutdown()


def test_uid_pins_an_empty_occurrence_to_its_own_row(tmp_path):
    engine = _engine(tmp_path)
    try:
        rows = [
            {"role": "user", "content": "q1" + PAD, "timestamp": 1.0, "message_uid": "a1"},
            {"role": "assistant", "content": "", "timestamp": 2.0, "message_uid": "e1"},
            {"role": "user", "content": "q2" + PAD, "timestamp": 3.0, "message_uid": "a2"},
            {"role": "assistant", "content": "", "timestamp": 4.0, "message_uid": "e2"},
        ]
        engine.ingest(rows)
        view = [dict(rows[3])]  # only the LATER empty reply is in view
        occurrences = [(0, engine._message_replay_identity(view[0], strip_carrier=False))]
        pinned, pins = engine._fork_uid_pinned_occurrences(occurrences, view)
        e1, e2 = [int(r["store_id"]) for r in engine._store.get_session_messages("S") if r["role"] == "assistant"]
        assert pinned[0][1] == ("fork-uid", "e2")
        assert set(pins) == {e2} and e1 not in pins  # the older empty reply can no longer be "shown" by it
        unbound = [(0, occurrences[0][1])]
        assert engine._fork_uid_pinned_occurrences(unbound, [dict(view[0], message_uid="zz")])[1] == {}
    finally:
        engine.shutdown()
