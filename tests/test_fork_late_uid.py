from tests.test_host_uid_shadow import _state_db, _engine, _bindings, _m, PAD


def test_late_uid_replay_binds(tmp_path):
    _state_db(tmp_path, [('S', None, None)])
    engine = _engine(tmp_path)
    try:
        row = _m('user', 'synthetic late' + PAD, 10.0)
        engine.ingest([row])
        engine.ingest([{**row, 'message_uid': 'late-uid'}])
        assert [(b[1], b[3]) for b in _bindings(engine)] == [('late-uid', 'canonical')]
    finally:
        engine.shutdown()


def test_late_uid_refuses_existing_binding_and_ambiguous_rows(tmp_path):
    _state_db(tmp_path, [('S', None, None)])
    engine = _engine(tmp_path)
    try:
        row = _m('user', 'synthetic late' + PAD, 10.0)
        engine.ingest([row])
        stored = engine._store.get_session_messages('S')[0]['store_id']
        lineage = engine._host_uid_lineage_key()[0]
        engine._store.add_host_uid_bindings(lineage, [(stored, 'prior', 'canonical', 'test')])
        engine.ingest([{**row, 'message_uid': 'late'}])
        assert not any(b[1] == 'late' and b[3] == 'canonical' for b in _bindings(engine))
    finally:
        engine.shutdown()


def test_late_uid_refuses_duplicate_occurrences(tmp_path):
    _state_db(tmp_path, [('S', None, None)])
    engine = _engine(tmp_path)
    try:
        row = _m('user', 'synthetic duplicate' + PAD, 10.0)
        engine.ingest([row])
        engine._store.append('S', row, conversation_id='conv')
        engine.ingest([{**row, 'message_uid': 'late'}])
        assert not _bindings(engine)
    finally:
        engine.shutdown()


def test_late_uid_refuses_changed_bytes(tmp_path):
    _state_db(tmp_path, [('S', None, None)])
    engine = _engine(tmp_path)
    try:
        row = _m('user', 'synthetic original' + PAD, 10.0)
        engine.ingest([row])
        sid = engine._store.get_session_messages('S')[0]['store_id']
        engine._store._conn.execute('UPDATE messages SET content=? WHERE store_id=?', ('synthetic altered', sid))
        engine._store._conn.commit()
        engine.ingest([{**row, 'message_uid': 'late'}])
        assert not _bindings(engine)
    finally:
        engine.shutdown()


def test_backfill_dry_run_and_apply(tmp_path):
    import json
    import sqlite3
    import subprocess
    import sys
    from pathlib import Path

    _state_db(tmp_path, [('S', None, None)])
    engine = _engine(tmp_path)
    try:
        engine.ingest([_m('user', 'synthetic backfill', 10.0)])
        engine._store._ensure_host_uid_schema()
    finally:
        engine.shutdown()
    state = sqlite3.connect(tmp_path / 'state.db')
    state.execute('CREATE TABLE messages (session_id TEXT, message_uid TEXT, role TEXT, content TEXT, tool_call_id TEXT, tool_calls TEXT)')
    state.execute('INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)', ('S', 'synthetic-uid', 'user', 'synthetic backfill', None, None))
    state.commit()
    state.close()
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'fork_backfill_uid_bindings.py'
    args = [sys.executable, str(script), '--db', str(tmp_path / 'lcm.db'), '--state-db', str(tmp_path / 'state.db')]
    dry = subprocess.run(args, capture_output=True, text=True, check=True)
    assert json.loads(dry.stdout)['candidates'] == 1
    conn = sqlite3.connect(tmp_path / 'lcm.db')
    assert conn.execute('SELECT count(*) FROM host_uid_bindings').fetchone()[0] == 0
    before = conn.execute('SELECT * FROM messages').fetchall()
    backup = sqlite3.connect(tmp_path / 'backup.db')
    conn.backup(backup)
    backup.close()
    conn.close()
    subprocess.run([*args, '--apply', '--backup-confirmed', str(tmp_path / 'backup.db')], check=True, capture_output=True)
    conn = sqlite3.connect(tmp_path / 'lcm.db')
    assert conn.execute('SELECT uid, kind, proof_kind FROM host_uid_bindings').fetchall() == [('synthetic-uid', 'canonical', 'backfill')]
    assert conn.execute('SELECT * FROM messages').fetchall() == before
    conn.close()


def test_late_uid_compaction_has_no_unpaired_rows(tmp_path, monkeypatch):
    import json
    import hermes_lcm.engine as module
    from hermes_lcm.tools import lcm_status

    monkeypatch.setattr(module, 'summarize_with_escalation', lambda **kwargs: ('Synthetic summary.', 1))
    _state_db(tmp_path, [('S', None, None)])
    engine = _engine(tmp_path)
    try:
        first = _m('user', 'synthetic first ' + 'word ' * 500, 10.0)
        others = [_m('assistant', f'synthetic {i} ' + 'word ' * 500, 11.0 + i, f'uid-{i}') for i in range(6)]
        engine.ingest([first, *others])
        view = [{**first, 'message_uid': 'late-uid'}, *others]
        engine.ingest(view)
        engine.compress(view, current_tokens=100000, force=True)
        audit = json.loads(lcm_status({}, engine=engine))['uid_coverage']
        assert audit['unpaired_count'] == 0
    finally:
        engine.shutdown()


def test_backfill_skips_ambiguous_and_bound_rows(tmp_path):
    import json
    import sqlite3
    import subprocess
    import sys
    from pathlib import Path

    _state_db(tmp_path, [('S', None, None)])
    engine = _engine(tmp_path)
    try:
        engine.ingest([_m('user', 'synthetic duplicate', 1), _m('user', 'synthetic bound', 2)])
        engine._store.append('S', _m('user', 'synthetic duplicate', 3), conversation_id='conv')
        bound = engine._store.get_session_messages('S')[1]['store_id']
        engine._store.add_host_uid_bindings(engine._host_uid_lineage_key()[0], [(bound, 'prior', 'canonical', 'test')])
    finally:
        engine.shutdown()
    state = sqlite3.connect(tmp_path / 'state.db')
    state.execute('CREATE TABLE messages (session_id TEXT, message_uid TEXT, role TEXT, content TEXT, tool_call_id TEXT, tool_calls TEXT)')
    state.executemany('INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)', [
        ('S', 'duplicate', 'user', 'synthetic duplicate', None, None),
        ('S', 'new-uid', 'user', 'synthetic bound', None, None),
    ])
    state.commit()
    state.close()
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'fork_backfill_uid_bindings.py'
    args = [sys.executable, str(script), '--db', str(tmp_path / 'lcm.db'), '--state-db', str(tmp_path / 'state.db')]
    dry = subprocess.run(args, capture_output=True, text=True, check=True)
    assert json.loads(dry.stdout)['candidates'] == 0
    denied = subprocess.run([*args, '--apply'], capture_output=True, text=True)
    assert denied.returncode != 0
