"""fork (10-26, default telegram session 20261024_165049_398f126f, third wedge): compaction stopped with
``leaf would end before an unresolved retained occurrence (store_id=22043 role=user kind=shown)``, and after the
restamped copies were cleared it stopped again at ``store_id=22445 role=assistant kind=shown``.

The turn's own ingest had missed a tool-call assistant row (state 128420, 05:35); its only stored copy came from a
later re-store (store 22675, 06:09), i.e. AFTER the stored rows of the host rows that follow it. The in-order store
mapper (``_get_store_id_map_for_messages``) mapped the host row to that late copy and moved its cursor past every
row in between, so the following host rows (22442..22674) stayed unmapped and the first of them was "shown" by an
unmapped view occurrence. The mapper now leaves such a row unmapped when its own uid is bound to the late copy and
a following host row's uid is bound to a stored row the jump would skip; the late copy stays stored. Rows without
that uid proof (a host-moved retained occurrence, #581/#1162 fixtures) keep the old mapping."""

from __future__ import annotations

import sqlite3

import pytest

import hermes_lcm.reconcile as reconcile_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

PAD = " alpha beta gamma delta" * 4


def _home(tmp_path):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    conn = sqlite3.connect(home / "state.db")
    conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, "
                 "end_reason TEXT, source TEXT, model_config TEXT)")
    conn.execute("INSERT OR REPLACE INTO sessions VALUES ('S', NULL, NULL, NULL, NULL)")
    conn.commit()
    conn.close()
    return str(home)


def _engine(tmp_path) -> LCMEngine:
    engine = LCMEngine(config=LCMConfig(fresh_tail_count=2, leaf_chunk_tokens=1,
                                        database_path=str(tmp_path / "lcm.db")), hermes_home=_home(tmp_path))
    engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
    return engine


def _call(n):
    return {"id": f"call_{n}", "type": "function", "function": {"name": "terminal", "arguments": f"{{\"n\":{n}}}"}}


def _turn():
    """user, then three tool-call rounds and a final reply (the 05:35 turn's shape), each with a host uid."""
    rows = [{"role": "user", "content": "delete that folder" + PAD, "timestamp": 10.0}]
    for n in range(1, 4):
        rows.append({"role": "assistant", "content": "", "tool_calls": [_call(n)], "timestamp": 10.0 + n})
        rows.append({"role": "tool", "content": f"{{\"output\": \"step {n}\"}}" + PAD, "tool_call_id": f"call_{n}",
                     "timestamp": 10.5 + n})
    rows.append({"role": "assistant", "content": "done" + PAD, "timestamp": 20.0})
    for i, row in enumerate(rows):
        row["message_uid"] = f"uid-{i}"
    return rows


def _plain(m):
    return {k: v for k, v in m.items() if k != "message_uid"}


def _store_with_late_copy(engine, *, bind=True):
    """Store the turn as the live session did: the first tool-call assistant missing in place, its only copy
    written after every other row of the turn, and each host uid canonically bound to its stored row."""
    turn = _turn()
    ids = {i: engine._store.append("S", _plain(m)) for i, m in enumerate(turn) if i != 1}
    ids[1] = engine._store.append("S", _plain(dict(turn[1], timestamp=99.0)))
    if bind:
        lineage = engine._host_uid_lineage_key()[0]
        assert lineage
        engine._store._ensure_host_uid_schema()
        conn = engine._store._conn
        cols = [r[1] for r in conn.execute("PRAGMA table_info(host_uid_bindings)")]
        for i, sid in ids.items():
            row = {"store_id": int(sid), "uid": turn[i]["message_uid"], "lineage_key": lineage, "kind": "canonical",
                   "binding_version": 1, "proof_kind": "test", "created_at": 0.0, "first_check": 0,
                   "disagree_seen": 0}
            conn.execute(f"INSERT INTO host_uid_bindings ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                         [row.get(c) for c in cols])
        conn.commit()
    return turn


def _mapped(engine, turn):
    mapping = engine._get_store_id_map_for_messages(turn)
    return [mapping.get(id(m)) for m in turn]


def test_late_copy_does_not_orphan_the_following_rows(tmp_path):
    engine = _engine(tmp_path)
    try:
        turn = _store_with_late_copy(engine)
        stored = engine._store.get_session_messages("S")
        assert stored[-1]["role"] == "assistant" and stored[-1]["tool_calls"]  # the late copy is last
        mapped = _mapped(engine, turn)
        assert mapped[0] is not None
        assert mapped[1] is None  # the late copy is left for a later pass, never jumped to
        assert all(sid is not None for sid in mapped[2:]), mapped
        assert mapped[2:] == sorted(mapped[2:])
    finally:
        engine.shutdown()


def test_late_copy_guard_off_restores_the_old_jump(tmp_path, monkeypatch):
    monkeypatch.setattr(reconcile_module, "_FORK_LATE_COPY_GUARD", False)
    engine = _engine(tmp_path)
    try:
        mapped = _mapped(engine, _store_with_late_copy(engine))
        assert mapped[1] is not None  # jumped to the late copy ...
        assert all(sid is None for sid in mapped[2:]), mapped  # ... and orphaned everything after it
    finally:
        engine.shutdown()


def test_without_uid_bindings_the_old_mapping_is_kept(tmp_path):
    """No uid proof (the #581/#1162 host-moved retained occurrence shape): the guard never fires."""
    engine = _engine(tmp_path)
    try:
        mapped = _mapped(engine, _store_with_late_copy(engine, bind=False))
    finally:
        engine.shutdown()
    assert mapped[1] is not None


def test_a_genuinely_later_row_still_maps_forward(tmp_path):
    engine = _engine(tmp_path)
    try:
        turn = _turn()
        for m in turn:
            engine._store.append("S", _plain(m))
        view = [turn[0], turn[-1]]  # the middle of the turn is not in view
        mapping = engine._get_store_id_map_for_messages(view)
        assert mapping.get(id(view[0])) is not None and mapping.get(id(view[1])) is not None
    finally:
        engine.shutdown()



def test_window_default_and_switch_names():
    import re
    from pathlib import Path
    src = Path(reconcile_module.__file__).read_text()
    assert re.search(r'os.environ.get\("LCM_FORK_LATE_COPY_GUARD", "on"\)', src)
    assert re.search(r'os.environ.get\("LCM_FORK_LATE_COPY_WINDOW", "16"\)', src)
