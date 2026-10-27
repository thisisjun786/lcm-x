"""Rewritten host UIDs retain occurrence identity across stored versions."""
import pytest

from tests.test_fork_late_copy_mapping import _engine, _plain, _mapped


def bind(engine, uid, sid, kind='canonical'):
    engine._store._ensure_host_uid_schema()
    conn = engine._store._conn
    cols = [r[1] for r in conn.execute('PRAGMA table_info(host_uid_bindings)')]
    row = {'store_id': sid, 'uid': uid, 'lineage_key': engine._host_uid_lineage_key()[0],
           'kind': kind, 'binding_version': 1, 'proof_kind': 'test', 'created_at': 0.0,
           'first_check': 0, 'disagree_seen': 0}
    conn.execute(f"INSERT INTO host_uid_bindings ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                 [row.get(c) for c in cols])
    conn.commit()


@pytest.mark.parametrize('role,old,new,extra', [
    ('user', '[Cron delivery: reminder]', '[Cron delivery: reminder]\n\nnext utterance', {}),
    ('assistant', 'partial response', 'partial response extended', {}),
    ('tool', 'full output', '[Externalized tool output: tool=terminal; tool_call_id=call_1; ref=abc]',
     {'tool_call_id': 'call_1'}),
])
def test_rewrite_without_stored_version_maps_to_original(tmp_path, role, old, new, extra):
    engine = _engine(tmp_path)
    try:
        original = {'role': role, 'content': old, 'message_uid': 'uid', **extra}
        sid = engine._store.append('S', _plain(original))
        bind(engine, 'uid', sid)
        current = {**original, 'content': new}
        assert _mapped(engine, [current]) == [sid]
    finally:
        engine.shutdown()


def test_unbound_late_copy_does_not_skip_following_uid(tmp_path):
    engine = _engine(tmp_path)
    try:
        first = {'role': 'user', 'content': 'old', 'message_uid': 'first'}
        following = {'role': 'assistant', 'content': 'next', 'message_uid': 'second'}
        first_id = engine._store.append('S', _plain(first))
        next_id = engine._store.append('S', _plain(following))
        changed = {**first, 'content': 'old appended'}
        engine._store.append('S', _plain(changed))
        bind(engine, 'first', first_id)
        bind(engine, 'second', next_id)
        assert _mapped(engine, [changed, following]) == [first_id, next_id]
    finally:
        engine.shutdown()


def test_rewritten_tool_coverage_includes_stored_full_output(tmp_path):
    engine = _engine(tmp_path)
    try:
        original = {'role': 'tool', 'content': 'synthetic complete tool output',
                    'tool_call_id': 'call_1', 'message_uid': 'tool-uid'}
        sid = engine._store.append('S', _plain(original))
        bind(engine, 'tool-uid', sid)
        current = {**original, 'content': '[Externalized tool output: tool=terminal; tool_call_id=call_1; ref=abc]'}
        leaf = engine._store_complete_input([current], {}, {id(current): sid}, [current], [current], 0, [], 10000)
        # None means the existing path uses the selected host row directly.
        effective_input = leaf if leaf is not None else [(current, [sid])]
        assert any(row.get('content') == original['content'] for row, _ids in effective_input)
    finally:
        engine.shutdown()
