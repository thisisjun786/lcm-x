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
    state.execute('CREATE TABLE messages (session_id TEXT, message_uid TEXT, role TEXT, content TEXT, tool_call_id TEXT, tool_calls TEXT, timestamp REAL)')
    state.execute('INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)', ('S', 'synthetic-uid', 'user', 'synthetic backfill', None, None, 10))
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
        # One UID-less later occurrence appears before the user at compaction.
        # The legacy positional fallback advances past the user's stored row;
        # a canonical late UID identifies it independently of view order.
        others[0].pop('message_uid')
        engine.ingest([first, *others])
        stamped = [{**first, 'message_uid': 'late-uid'}, *others]
        engine.ingest(stamped)
        # A later host view may omit the source timestamp; the UID must still pair it.
        view = [others[0], {k: v for k, v in stamped[0].items() if k != 'timestamp'}, *others[1:]]
        engine.compress(view, current_tokens=100000, force=True)
        audit = json.loads(lcm_status({}, engine=engine))['uid_coverage']
        assert audit['context_shrank'] is True
        assert audit['unpaired_count'] == 0, audit
        assert audit['unpaired'] == []
        assert len(audit['paired_store_ids']) == 7
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


def test_backfill_counts_uidless_host_occurrences(tmp_path):
    import json
    import sqlite3
    import subprocess
    import sys
    from pathlib import Path

    _state_db(tmp_path, [('S', None, None)])
    engine = _engine(tmp_path)
    try:
        engine.ingest([_m('user', 'synthetic same', 10)])
        engine._store._ensure_host_uid_schema()
    finally:
        engine.shutdown()
    state = sqlite3.connect(tmp_path / 'state.db')
    state.execute('CREATE TABLE messages (session_id TEXT, message_uid TEXT, role TEXT, content TEXT, tool_call_id TEXT, tool_calls TEXT)')
    state.executemany('INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)', [
        ('S', None, 'user', 'synthetic same', None, None),
        ('S', 'second-occurrence', 'user', 'synthetic same', None, None),
    ])
    state.commit()
    state.close()
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'fork_backfill_uid_bindings.py'
    result = subprocess.run([sys.executable, str(script), '--db', str(tmp_path / 'lcm.db'),
                             '--state-db', str(tmp_path / 'state.db')], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['candidates'] == 0


def test_backfill_does_not_conflate_tool_names(tmp_path):
    import json
    import sqlite3
    import subprocess
    import sys
    from pathlib import Path

    _state_db(tmp_path, [('S', None, None)])
    engine = _engine(tmp_path)
    try:
        engine.ingest([_m('tool', 'synthetic shared output', 10, tool_call_id='call-synthetic', tool_name='terminal')])
        engine._store._ensure_host_uid_schema()
    finally:
        engine.shutdown()
    state = sqlite3.connect(tmp_path / 'state.db')
    state.execute('CREATE TABLE messages (session_id TEXT, message_uid TEXT, role TEXT, content TEXT, tool_call_id TEXT, tool_calls TEXT, tool_name TEXT)')
    state.execute('INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)',
                  ('S', 'synthetic-browser', 'tool', 'synthetic shared output', 'call-synthetic', None, 'browser'))
    state.commit()
    state.close()
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'fork_backfill_uid_bindings.py'
    result = subprocess.run([sys.executable, str(script), '--db', str(tmp_path / 'lcm.db'),
                             '--state-db', str(tmp_path / 'state.db')], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['candidates'] == 0


def test_backfill_skips_lossy_redactions(tmp_path):
    import json
    import sqlite3
    import subprocess
    import sys
    from pathlib import Path

    _state_db(tmp_path, [('S', None, None)])
    redaction = '[LCM sensitive redaction: name=password_assignment; chars=8; bytes=8]'
    call = [{'id': 'synthetic-call', 'type': 'function', 'function': {'name': 'synthetic', 'arguments': redaction}}]
    engine = _engine(tmp_path)
    try:
        engine.ingest([_m('user', redaction, 10), _m('assistant', 'synthetic call', 11, tool_calls=call)])
        engine._store._ensure_host_uid_schema()
    finally:
        engine.shutdown()
    state = sqlite3.connect(tmp_path / 'state.db')
    state.execute('CREATE TABLE messages (session_id TEXT, message_uid TEXT, role TEXT, content TEXT, tool_call_id TEXT, tool_calls TEXT)')
    state.executemany('INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)', [
        ('S', 'synthetic-user', 'user', redaction, None, None),
        ('S', 'synthetic-call', 'assistant', 'synthetic call', None, json.dumps(call)),
    ])
    state.commit()
    state.close()
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'fork_backfill_uid_bindings.py'
    args = [sys.executable, str(script), '--db', str(tmp_path / 'lcm.db'), '--state-db', str(tmp_path / 'state.db')]
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['candidates'] == 0


import pytest


@pytest.mark.parametrize('case,expected', [
    ('positive', 1), ('uidless', 0), ('tool-name', 0), ('redacted-content', 0),
    ('redacted-arguments', 0), ('timestamp', 0), ('pruned', 0), ('persisted', 0),
    ('todo', 0), ('model-switch', 0), ('unparseable', 0), ('missing-tool-name', 0),
    ('summary-carrier', 0), ('out-of-band', 0), ('no-observed-at', 1),
])
def test_backfill_identity_table(tmp_path, case, expected):
    import json
    import sqlite3
    import subprocess
    import sys
    from pathlib import Path
    from hermes_lcm import reconcile

    _state_db(tmp_path, [('S', None, None)])
    row = dict(role='user', content='synthetic identity', tool_call_id=None, tool_calls=None, tool_name=None)
    redaction = '[LCM sensitive redaction: name=password_assignment; chars=8; bytes=8]'
    if case in ('tool-name', 'missing-tool-name'):
        row.update(role='tool', tool_call_id='synthetic-call', tool_name='terminal')
    if case == 'redacted-content':
        row['content'] = redaction
    if case == 'redacted-arguments':
        row.update(role='assistant', tool_calls=[{'id': 'synthetic', 'function': {'arguments': redaction}}])
    if case == 'pruned':
        row['content'] = reconcile._HERMES_PRUNED_TOOL_PLACEHOLDER
    if case == 'persisted':
        row['content'] = '<persisted-output>\nThis tool result was too large (99 characters, 1 KB).\nFull output saved to: synthetic\n</persisted-output>'
    if case == 'todo':
        row['content'] = 'synthetic\n\n' + reconcile._PRESERVED_TODO_CONTEXT_PREFIX
    if case == 'model-switch':
        row['content'] = reconcile._MODEL_SWITCH_NOTIFICATION_PREFIX + 'synthetic] text'
    if case == 'out-of-band':
        row['content'] = 'synthetic ' + reconcile._OOB_MESSAGE_BLOCK_MARKER + ']text[/OUT-OF-BAND USER MESSAGE]'
    if case == 'summary-carrier':
        row['content'] = '[Recent Summary (d0, node 1)]\nsynthetic\n\nreal user'
    engine = _engine(tmp_path)
    try:
        engine._store.append('S', {**row, 'timestamp': 10}, conversation_id='conv')
        engine._store._ensure_host_uid_schema()
        if case == 'no-observed-at':
            engine._store._conn.execute('UPDATE messages SET observed_at = NULL')
            engine._store._conn.commit()
    finally:
        engine.shutdown()
    state = sqlite3.connect(tmp_path / 'state.db')
    state.execute('CREATE TABLE messages (session_id TEXT, message_uid TEXT, role TEXT, content TEXT, tool_call_id TEXT, tool_calls TEXT, tool_name TEXT, timestamp REAL)')
    calls = json.dumps(row['tool_calls']) if row['tool_calls'] else None
    host_name = 'browser' if case == 'tool-name' else None if case == 'missing-tool-name' else row['tool_name']
    if case == 'unparseable':
        calls = '{broken'
    values = ('S', 'synthetic-uid', row['role'], row['content'], row['tool_call_id'], calls, host_name,
              20 if case == 'timestamp' else 10)
    state.execute('INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)', values)
    if case == 'uidless':
        state.execute('INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                      (values[0], None, *values[2:]))
    state.commit()
    state.close()
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'fork_backfill_uid_bindings.py'
    result = subprocess.run([sys.executable, str(script), '--db', str(tmp_path / 'lcm.db'),
                             '--state-db', str(tmp_path / 'state.db')], capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    assert data['candidates'] == expected
    if case == 'positive':
        assert data['timestamp_matched'] == 1 and data['unique_without_timestamp'] == 0
    if case == 'no-observed-at':
        assert data['timestamp_matched'] == 0 and data['unique_without_timestamp'] == 1
