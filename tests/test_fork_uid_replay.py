"""fork: a gateway restart replays assistant/tool rows WITHOUT their timestamp (gateway/run.py strips it from
tool rows and keeps it only on user rows). #436 cannot recognise them by stamp, so before this fix every
restart / manual /compress re-stored the whole history as NULL-stamp copies, and those copies later wedged
compaction ("unresolved retained occurrence"). The host message_uid binding proves the row is a replay."""

from __future__ import annotations

import sqlite3

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

PAD = " alpha beta gamma delta" * 4


def _state_db(tmp_path) -> None:
    conn = sqlite3.connect(tmp_path / "state.db")
    conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, "
                 "end_reason TEXT, source TEXT, model_config TEXT)")
    conn.execute("INSERT OR REPLACE INTO sessions VALUES ('S', NULL, NULL, NULL, NULL)")
    conn.commit()
    conn.close()


def _engine(tmp_path) -> LCMEngine:
    engine = LCMEngine(config=LCMConfig(fresh_tail_count=2, leaf_chunk_tokens=1,
                                        database_path=str(tmp_path / "lcm.db")))
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


def _gateway_replay(history):
    """What gateway/run.py hands the engine after a restart: user rows keep their stamp, the rest lose it."""
    out = []
    for message in history:
        copy = dict(message)
        if copy["role"] != "user":
            copy.pop("timestamp", None)
        out.append(copy)
    return out


def _count(engine) -> int:
    return engine._store._conn.execute("SELECT count(*) FROM messages WHERE session_id='S'").fetchone()[0]


def test_restart_replay_without_stamps_is_not_stored_again(tmp_path):
    _state_db(tmp_path)
    history = _history()
    first = _engine(tmp_path)
    try:
        first.ingest(history)
        assert _count(first) == 4
    finally:
        first.shutdown()
    restarted = _engine(tmp_path)  # a fresh process: cursor 0, no cached prefix
    try:
        # a host row LCM never saw sits in front, so the ordered prefix proves nothing (the real failure shape)
        lead = {"role": "user", "content": "[Recent Summary] carried" + PAD, "timestamp": 9.0, "message_uid": "u-0"}
        view = [lead, *_gateway_replay(history), {"role": "user", "content": "next" + PAD,
                                                  "timestamp": 13.0, "message_uid": "u-5"}]
        restarted.ingest(view)
        contents = [r["content"] for r in restarted._store.get_session_messages("S")]
        assert contents.count("two files" + PAD) == 1
        assert contents.count("{\"output\": \"a.txt b.txt\"}" + PAD) == 1
        assert "next" + PAD in contents
    finally:
        restarted.shutdown()


def test_same_uid_with_new_bytes_is_still_stored(tmp_path):
    _state_db(tmp_path)
    history = _history()
    first = _engine(tmp_path)
    try:
        first.ingest(history)
    finally:
        first.shutdown()
    restarted = _engine(tmp_path)
    try:
        lead = {"role": "user", "content": "[Recent Summary] carried" + PAD, "timestamp": 9.0, "message_uid": "u-0"}
        edited = _gateway_replay(history)
        edited[3] = {**edited[3], "content": "three files" + PAD}  # same uid u-4, new bytes
        restarted.ingest([lead, *edited])
        contents = [r["content"] for r in restarted._store.get_session_messages("S")]
        assert "three files" + PAD in contents
        assert contents.count("two files" + PAD) == 1
    finally:
        restarted.shutdown()


def test_off_mode_keeps_the_old_behaviour(tmp_path, monkeypatch):
    _state_db(tmp_path)
    history = _history()
    first = _engine(tmp_path)
    try:
        first.ingest(history)
    finally:
        first.shutdown()
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    restarted = _engine(tmp_path)
    try:
        assert restarted._fork_uid_replayed_indexes(_gateway_replay(history), 0) == set()
    finally:
        restarted.shutdown()
