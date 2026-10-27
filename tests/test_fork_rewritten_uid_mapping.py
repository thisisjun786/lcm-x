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


def test_newest_bound_version_wins_even_when_old_bytes_are_shown(tmp_path):
    engine = _engine(tmp_path)
    try:
        old = {'role': 'user', 'content': 'synthetic original', 'message_uid': 'versioned'}
        new = {**old, 'content': 'synthetic revised'}
        old_id = engine._store.append('S', _plain(old))
        new_id = engine._store.append('S', _plain(new))
        bind(engine, 'versioned', old_id)
        bind(engine, 'versioned', new_id, 'version')
        assert _mapped(engine, [old]) == [new_id]
        assert len(engine._store.get_session_messages('S')) == 2
    finally:
        engine.shutdown()


def test_reordered_bound_rows_do_not_move_the_fallback_cursor(tmp_path):
    engine = _engine(tmp_path)
    try:
        rows = [{'role': 'user', 'content': f'synthetic row {i}', 'message_uid': f'uid-{i}'} for i in range(4)]
        ids = [engine._store.append('S', _plain(row)) for row in rows]
        for row, sid in zip(rows, ids):
            bind(engine, row['message_uid'], sid)
        view = [rows[i] for i in (2, 0, 3, 1)]
        assert _mapped(engine, view) == [ids[i] for i in (2, 0, 3, 1)]
    finally:
        engine.shutdown()


def test_uid_without_binding_uses_content_fallback(tmp_path):
    engine = _engine(tmp_path)
    try:
        row = {'role': 'user', 'content': 'synthetic unbound', 'message_uid': 'unbound'}
        sid = engine._store.append('S', _plain(row))
        assert _mapped(engine, [row]) == [sid]
    finally:
        engine.shutdown()


def test_uid_does_not_borrow_another_profiles_binding(tmp_path):
    engine = _engine(tmp_path)
    try:
        row = {'role': 'user', 'content': 'synthetic original', 'message_uid': 'shared'}
        sid = engine._store.append('S', _plain(row))
        bind(engine, 'shared', sid)
        engine._store._conn.execute("UPDATE host_uid_bindings SET lineage_key = 'other:S'")
        engine._store._conn.commit()
        assert _mapped(engine, [{**row, 'content': 'synthetic changed'}]) == [None]
    finally:
        engine.shutdown()


def test_reordered_leaf_covers_paired_set(tmp_path):
    engine = _engine(tmp_path)
    try:
        rows = [{'role': 'user', 'content': f'synthetic row {i}', 'message_uid': f'uid-{i}'} for i in range(4)]
        ids = [engine._store.append('S', _plain(row)) for row in rows]
        for row, sid in zip(rows, ids):
            bind(engine, row['message_uid'], sid)
        view = [rows[i] for i in (2, 0, 3, 1)]
        mapping = engine._get_store_id_map_for_messages(view)
        result = engine._store_complete_input(view, {}, mapping, view, view, 0, [], 10000)
        assert result != [], engine._store_complete_empty_reason
        effective = result if result is not None else [(row, []) for row in view]
        covered = {sid for row, claims in effective for sid in [*claims, *([mapping[id(row)]] if id(row) in mapping else [])]}
        assert covered == set(ids)
    finally:
        engine.shutdown()


def test_newest_version_does_not_reclaim_historical_bytes(tmp_path):
    engine = _engine(tmp_path)
    try:
        old = {'role': 'user', 'content': 'synthetic original', 'message_uid': 'versioned'}
        new = {**old, 'content': 'synthetic revised'}
        old_id = engine._store.append('S', _plain(old))
        new_id = engine._store.append('S', _plain(new))
        bind(engine, 'versioned', old_id)
        bind(engine, 'versioned', new_id, 'version')
        mapping = engine._get_store_id_map_for_messages([new])
        result = engine._store_complete_input([new], {}, mapping, [new], [new], 0, [], 10000)
        effective = result if result is not None else [(new, [])]
        covered = {sid for row, claims in effective for sid in [*claims, *([mapping[id(row)]] if id(row) in mapping else [])]}
        assert covered == {new_id}
        assert old_id in engine._store_complete_excluded
    finally:
        engine.shutdown()


def test_new_version_after_retained_neighbor_can_form_leaf(tmp_path):
    engine = _engine(tmp_path)
    try:
        old = {'role': 'user', 'content': 'synthetic original', 'message_uid': 'versioned'}
        retained = {'role': 'assistant', 'content': 'synthetic fresh reply', 'message_uid': 'retained'}
        new = {**old, 'content': 'synthetic revised'}
        old_id = engine._store.append('S', _plain(old))
        retained_id = engine._store.append('S', _plain(retained))
        new_id = engine._store.append('S', _plain(new))
        bind(engine, 'versioned', old_id)
        bind(engine, 'retained', retained_id)
        bind(engine, 'versioned', new_id, 'version')
        view = [new, retained]
        mapping = engine._get_store_id_map_for_messages(view)
        result = engine._store_complete_input([new], {}, mapping, view, [new], 0, [], 10000)
        assert result != [], engine._store_complete_blocked
        effective = result if result is not None else [(new, [])]
        covered = {sid for row, claims in effective for sid in [*claims, *([mapping[id(row)]] if id(row) in mapping else [])]}
        assert covered == {new_id}
        assert retained_id not in engine._store_complete_excluded
    finally:
        engine.shutdown()


def test_unpaired_row_is_not_consumed_with_later_paired_row(tmp_path):
    engine = _engine(tmp_path)
    try:
        unknown = {'role': 'user', 'content': 'synthetic unavailable source', 'message_uid': 'unknown'}
        paired = {'role': 'assistant', 'content': 'synthetic stored reply', 'message_uid': 'paired'}
        sid = engine._store.append('S', _plain(paired))
        bind(engine, 'paired', sid)
        engine._compress_run.store_ids_by_message_id = {id(paired): sid}
        engine._store_complete_cut = True
        consumed = engine._leaf_consumption([unknown, paired], [paired], {}, [(paired, [sid])], {id(paired): [sid]}, set())
        assert unknown not in consumed[2]
    finally:
        engine.shutdown()


def _publish(engine, sid, uid, excluded=()):
    from hermes_lcm.dag import SummaryNode
    engine._uid_source_bindings = {sid: uid}
    engine._store_complete_copy_exclusion_proofs = {}
    frontier = engine._lifecycle.get_by_conversation('conv').current_frontier_store_id
    node = SummaryNode(session_id='S', summary='Synthetic summary', depth=0,
                       source_ids=[sid], source_type='messages', token_count=3, source_token_count=100)
    engine._publish_leaf_node(node, level=1, model='synthetic', expected_frontier=frontier,
                              consumed_store_ids=[sid], publication_excluded_store_ids=list(excluded),
                              filter_exclusion_proofs={}, carried_ranges=[])
    return node


def test_sparse_publication_preserves_gap_across_restart(tmp_path):
    engine = _engine(tmp_path)
    try:
        ids = []
        for i in range(3):
            sid = engine._store.append('S', {'role': 'user', 'content': f'synthetic {i}'})
            bind(engine, f'uid-{i}', sid)
            ids.append(sid)
        _publish(engine, ids[2], 'uid-2')
        assert engine._lifecycle.get_by_conversation('conv').current_frontier_store_id == 0
    finally:
        engine.shutdown()
    engine = _engine(tmp_path)
    try:
        assert engine._last_compacted_store_id == 0
        _publish(engine, ids[0], 'uid-0')
        assert engine._lifecycle.get_by_conversation('conv').current_frontier_store_id == ids[0]
        _publish(engine, ids[1], 'uid-1')
        assert engine._lifecycle.get_by_conversation('conv').current_frontier_store_id >= ids[1]
        covered = [sid for node in engine._dag.get_session_nodes('S') for sid in node.source_ids]
        assert sorted(covered) == ids
        assert len(engine._store.get_session_messages('S')) == 3
    finally:
        engine.shutdown()


def test_sparse_publication_rechecks_latest_uid_atomically(tmp_path):
    from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError
    engine = _engine(tmp_path)
    try:
        first = engine._store.append('S', {'role': 'user', 'content': 'synthetic first'})
        bind(engine, 'version', first)
        second = engine._store.append('S', {'role': 'user', 'content': 'synthetic second'})
        bind(engine, 'version', second, 'version')
        with pytest.raises(LifecyclePublicationConflictError, match='binding changed'):
            _publish(engine, first, 'version')
        assert engine._dag.get_session_nodes('S') == []
        assert engine._lifecycle.get_by_conversation('conv').current_frontier_store_id == 0
    finally:
        engine.shutdown()


def test_superseded_history_does_not_match_uidless_fallback(tmp_path):
    engine = _engine(tmp_path)
    try:
        old = {'role': 'user', 'content': 'synthetic old'}
        first = engine._store.append('S', old)
        second = engine._store.append('S', {**old, 'content': 'synthetic new'})
        bind(engine, 'version', first)
        bind(engine, 'version', second, 'version')
        assert _mapped(engine, [old]) == [None]
    finally:
        engine.shutdown()


def test_reordered_compaction_shrinks_and_keeps_fresh_tail(tmp_path, monkeypatch):
    import hermes_lcm.engine as module
    from hermes_lcm.tokens import count_messages_tokens
    monkeypatch.setattr(module, 'summarize_with_escalation', lambda **kwargs: ('Synthetic past turns.', 1))
    engine = _engine(tmp_path)
    try:
        rows = [{'role': 'assistant', 'content': f'synthetic {i} ' + 'word ' * 500,
                 'message_uid': f'uid-{i}'} for i in range(6)]
        ids = []
        for row in rows:
            sid = engine._store.append('S', _plain(row))
            bind(engine, row['message_uid'], sid)
            ids.append(sid)
        view = [rows[i] for i in (2, 0, 3, 1, 4, 5)]
        engine._ingest_cursor = len(view)
        result = engine.compress(view, current_tokens=100000, force=True)
        assert count_messages_tokens(result) < count_messages_tokens(view)
        assert all(any(row['content'].strip() in m.get('content', '') for m in result) for row in rows[-2:]), [(m.get('role'), len(m.get('content', '')), m.get('message_uid')) for m in result]
        nodes = engine._dag.get_session_nodes('S')
        covered = {sid for node in nodes if node.source_type == 'messages' for sid in node.source_ids}
        assert covered and covered <= set(ids[:4])
        assert len(engine._store.get_session_messages('S')) == 6
    finally:
        engine.shutdown()


@pytest.mark.parametrize('with_uid', [True, False])
def test_unpaired_row_remains_visible_while_later_rows_compact(tmp_path, monkeypatch, caplog, with_uid):
    import json
    import logging
    import hermes_lcm.engine as module
    from hermes_lcm.tools import lcm_status
    monkeypatch.setattr(module, 'summarize_with_escalation', lambda **kwargs: ('Synthetic earlier turns.', 1))
    engine = _engine(tmp_path)
    try:
        unknown = {'role': 'assistant', 'content': 'UNPAIRED_SENTINEL ' + 'word ' * 500,
                   'message_uid': 'unknown'}
        rows = [{'role': 'assistant', 'content': f'synthetic {i} ' + 'word ' * 500,
                 'message_uid': f'uid-{i}'} for i in range(6)]
        if not with_uid:
            unknown.pop('message_uid')
            for row in rows:
                row.pop('message_uid')
        for row in rows:
            sid = engine._store.append('S', _plain(row))
            if with_uid:
                bind(engine, row['message_uid'], sid)
        view = [unknown, *rows]
        engine._ingest_cursor = len(view)
        with caplog.at_level(logging.INFO, logger='hermes_lcm.compaction'):
            result = engine.compress(view, current_tokens=100000, force=True)
        assert any(unknown['content'].strip() in m.get('content', '') for m in result)
        assert engine._dag.get_session_nodes('S')
        audit = json.loads(lcm_status({}, engine=engine))['uid_coverage']
        assert audit['unpaired_count'] == 1
        assert audit['unpaired'] == [{'store_id': None, 'uid': 'unknown' if with_uid else None}]
        assert 'UNPAIRED_SENTINEL' not in json.dumps(audit)
        stops = [r.getMessage() for r in caplog.records if 'LCM compaction stop:' in r.getMessage()]
        assert stops and 'unpaired_count=1' in stops[-1]
        assert 'UNPAIRED_SENTINEL' not in stops[-1]
    finally:
        engine.shutdown()


def test_compaction_advances_past_new_version_without_consuming_retained_gap(tmp_path, monkeypatch):
    import hermes_lcm.engine as module
    from hermes_lcm.tokens import count_messages_tokens
    monkeypatch.setattr(module, 'summarize_with_escalation', lambda **kwargs: ('Synthetic earlier turn.', 1))
    engine = _engine(tmp_path)
    try:
        old = {'role': 'user', 'content': 'synthetic old ' + 'word ' * 500, 'message_uid': 'version'}
        retained = {'role': 'user', 'content': 'synthetic fresh objective', 'message_uid': 'fresh'}
        new = {**old, 'content': old['content'] + '\n\nsynthetic appended input'}
        tail = {'role': 'assistant', 'content': 'synthetic fresh reply', 'message_uid': 'tail'}
        ids = [engine._store.append('S', _plain(m)) for m in (old, retained, new, tail)]
        for m, sid, kind in zip((old, retained, new, tail), ids, ('canonical', 'canonical', 'version', 'canonical')):
            bind(engine, m['message_uid'], sid, kind)
        view = [new, retained, tail]
        engine._ingest_cursor = len(view)
        result = engine.compress(view, current_tokens=100000, force=True)
        assert count_messages_tokens(result) < count_messages_tokens(view)
        assert any(retained['content'] in m.get('content', '') for m in result)
        covered = {sid for n in engine._dag.get_session_nodes('S') if n.source_type == 'messages' for sid in n.source_ids}
        assert covered == {ids[2]}
        assert engine._lifecycle.get_by_conversation('conv').current_frontier_store_id == ids[0]
        before_restart = engine._store.get_session_messages('S')
        assert len(before_restart) == 4
        engine.shutdown()
        engine = _engine(tmp_path)
        engine.ingest([dict(message) for message in result])
        assert engine._store.get_session_messages('S') == before_restart
        assert ids[1] in engine._get_store_id_map_for_messages(result).values()
        _publish(engine, ids[1], 'fresh')
        sources = [sid for node in engine._dag.get_session_nodes('S')
                   if node.source_type == 'messages' for sid in node.source_ids]
        assert sorted(sources) == sorted([ids[1], ids[2]])
        assert engine._store.get_session_messages('S') == before_restart
    finally:
        engine.shutdown()


def test_nonshrinking_uid_leaf_does_not_claim_progress(tmp_path, monkeypatch, caplog):
    import logging
    from hermes_lcm.tokens import count_messages_tokens
    engine = _engine(tmp_path)
    try:
        rows = [{'role': 'user' if i % 2 == 0 else 'assistant',
                 'content': f'synthetic {i} ' + 'word ' * 50, 'message_uid': f'uid-{i}'} for i in range(6)]
        for row in rows:
            sid = engine._store.append('S', _plain(row))
            bind(engine, row['message_uid'], sid)
        monkeypatch.setattr(engine, '_summarize_leaf_chunk_with_rescue',
                            lambda chunk, **kwargs: (chunk, count_messages_tokens(chunk), 'expansion ' * 10000, 1, 0))
        engine._ingest_cursor = len(rows)
        with caplog.at_level(logging.INFO, logger='hermes_lcm.compaction'):
            engine.compress(rows, current_tokens=100000, force=True)
        assert engine.get_status()['uid_coverage']['context_shrank'] is False
        assert engine.last_compression_status == 'noop'
        assert len(engine._store.get_session_messages('S')) == len(rows)
        stops = [r.getMessage() for r in caplog.records if 'LCM compaction stop:' in r.getMessage()]
        assert stops and 'progress=none' in stops[-1]
    finally:
        engine.shutdown()


def test_quarantined_version_cannot_override_owned_pairing(tmp_path):
    engine = _engine(tmp_path)
    try:
        row = {'role': 'user', 'content': 'synthetic owned', 'message_uid': 'version'}
        first = engine._store.append('S', _plain(row))
        second = engine._store.append('S', {**_plain(row), 'content': 'synthetic quarantined'})
        bind(engine, 'version', first)
        bind(engine, 'version', second, 'version')
        engine._store._conn.execute('UPDATE messages SET session_id = ? WHERE store_id = ?', ('quarantine', second))
        engine._store._conn.commit()
        assert _mapped(engine, [row]) == [first]
        _publish(engine, first, 'version')
        assert {sid for n in engine._dag.get_session_nodes('S') for sid in n.source_ids} == {first}
    finally:
        engine.shutdown()


def test_summary_header_growth_is_not_reported_as_progress(tmp_path, monkeypatch, caplog):
    import logging
    from hermes_lcm.tokens import count_messages_tokens
    engine = _engine(tmp_path)
    try:
        rows = [{'role': role, 'content': f'synthetic {i} ' + 'word ' * 50, 'message_uid': f'uid-{i}'}
                for i, role in enumerate(('assistant', 'user', 'assistant'))]
        for row in rows:
            sid = engine._store.append('S', _plain(row))
            bind(engine, row['message_uid'], sid)
        monkeypatch.setattr(engine, '_summarize_leaf_chunk_with_rescue',
                            lambda chunk, **kwargs: (chunk, count_messages_tokens(chunk), chunk[0]['content'], 1, 0))
        engine._ingest_cursor = len(rows)
        with caplog.at_level(logging.INFO, logger='hermes_lcm.compaction'):
            result = engine.compress(rows, current_tokens=100000, force=True)
        assert count_messages_tokens(result) >= count_messages_tokens(rows)
        stops = [r.getMessage() for r in caplog.records if 'LCM compaction stop:' in r.getMessage()]
        assert stops and 'progress=none' in stops[-1]
        assert engine.last_compression_status != 'compacted'
    finally:
        engine.shutdown()


def test_late_tool_call_is_compacted_with_its_result(tmp_path, monkeypatch):
    import hermes_lcm.engine as module
    from tests.test_fork_late_copy_mapping import _store_with_late_copy
    from hermes_lcm.tokens import count_messages_tokens
    monkeypatch.setattr(module, 'summarize_with_escalation', lambda **kwargs: ('Synthetic tool rounds.', 1))
    engine = _engine(tmp_path)
    try:
        view = _store_with_late_copy(engine)
        before = engine._store.get_session_messages('S')
        mapping = engine._get_store_id_map_for_messages(view)
        engine._ingest_cursor = len(view)
        result = engine.compress(view, current_tokens=100000, force=True)
        leaves = [n for n in engine._dag.get_session_nodes('S') if n.source_type == 'messages']
        owner = [n for n in leaves if mapping[id(view[1])] in n.source_ids]
        assert len(owner) == 1
        assert mapping[id(view[2])] in owner[0].source_ids
        assert count_messages_tokens(result) < count_messages_tokens(view)
        assert engine._store.get_session_messages('S') == before
        for n in leaves:
            for i in (1, 3, 5):
                assert (mapping[id(view[i])] in n.source_ids) == (mapping[id(view[i + 1])] in n.source_ids)
    finally:
        engine.shutdown()


def test_early_summary_carrier_and_reordered_rows_compact_without_restoring_summary(tmp_path, monkeypatch):
    import hermes_lcm.engine as module
    from hermes_lcm.tokens import count_messages_tokens
    monkeypatch.setattr(module, 'summarize_with_escalation', lambda **kwargs: ('Synthetic subsequent turns.', 1))
    engine = _engine(tmp_path)
    try:
        prior = engine._store.append('S', {'role': 'assistant', 'content': 'synthetic prior history'})
        bind(engine, 'prior', prior)
        _publish(engine, prior, 'prior')
        rows = [{'role': 'user' if i % 2 == 0 else 'assistant',
                 'content': f'synthetic {i} ' + 'word ' * 100, 'message_uid': f'uid-{i}'} for i in range(6)]
        ids = []
        for row in rows:
            sid = engine._store.append('S', _plain(row))
            bind(engine, row['message_uid'], sid)
            ids.append(sid)
        assembled = engine._assemble_context(None, rows)
        carrier = assembled[0]
        assert engine._generated_context_carrier_remainder(carrier) == rows[0]['content']
        view = [carrier, rows[3], rows[2], rows[1], rows[4], rows[5]]
        mapping = engine._get_store_id_map_for_messages(view)
        assert [mapping[id(row)] for row in view[1:]] == [ids[i] for i in (3, 2, 1, 4, 5)]
        before = engine._store.get_session_messages('S')
        engine._ingest_cursor = len(view)
        result = engine.compress(view, current_tokens=100000, force=True)
        covered = {sid for n in engine._dag.get_session_nodes('S') if n.source_type == 'messages' for sid in n.source_ids}
        assert covered == {prior, *ids[:4]}
        assert count_messages_tokens(result) < count_messages_tokens(view)
        assert all(any(row['content'].strip() in item.get('content', '') for item in result) for row in rows[-2:])
        assert engine._store.get_session_messages('S') == before
    finally:
        engine.shutdown()


def test_uid_mapping_fetches_payloads_only_for_present_uids(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    try:
        rows = [{'role': 'assistant', 'content': f'synthetic {i} ' + 'word ' * 100,
                 'message_uid': f'uid-{i}'} for i in range(20)]
        ids = []
        for row in rows:
            sid = engine._store.append('S', _plain(row))
            ids.append(sid)
            bind(engine, row['message_uid'], sid)
        fetch = engine._store.get_batch
        fetched = []
        def record(wanted):
            fetched.extend(wanted)
            return fetch(wanted)
        monkeypatch.setattr(engine._store, 'get_batch', record)
        pairs, superseded = engine._uid_pairing_rows([rows[-1]])
        assert set(pairs) == {'uid-19'} and not superseded
        assert fetched == [ids[-1]]
    finally:
        engine.shutdown()


def test_bound_uid_pairing_uses_existing_evidence_when_capture_is_off(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    try:
        row = {'role': 'user', 'content': 'synthetic original', 'message_uid': 'bound'}
        sid = engine._store.append('S', _plain(row))
        bind(engine, 'bound', sid)
        monkeypatch.setenv('LCM_HOST_MESSAGE_UID', 'off')
        assert _mapped(engine, [{**row, 'content': 'synthetic rewritten'}]) == [sid]
        assert engine._store._conn.execute('SELECT count(*) FROM host_uid_bindings').fetchone()[0] == 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize('bound_first', [True, False])
@pytest.mark.parametrize('fallback_uid', [None, 'unbound'])
def test_bound_occurrence_does_not_consume_fallback_identity_count(tmp_path, bound_first, fallback_uid):
    engine = _engine(tmp_path)
    try:
        bound = {'role': 'assistant', 'content': 'synthetic repeated bytes', 'message_uid': 'bound'}
        fallback = {'role': 'assistant', 'content': bound['content']}
        if fallback_uid:
            fallback['message_uid'] = fallback_uid
        first = engine._store.append('S', _plain(bound))
        second = engine._store.append('S', _plain(fallback))
        bind(engine, 'bound', first)
        view = [bound, fallback] if bound_first else [fallback, bound]
        assert _mapped(engine, view) == ([first, second] if bound_first else [second, first])
    finally:
        engine.shutdown()


def test_version_exclusion_is_rechecked_for_content_fallback_publication(tmp_path):
    from hermes_lcm.dag import SummaryNode
    from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError
    engine = _engine(tmp_path)
    try:
        old = engine._store.append('S', {'role': 'user', 'content': 'synthetic old'})
        new = engine._store.append('S', {'role': 'user', 'content': 'synthetic new'})
        bind(engine, 'version', old)
        bind(engine, 'version', new, 'version')
        engine._uid_source_bindings = {}
        engine._uid_sparse_coverage = False
        engine._uid_superseded_store_ids = {old}
        engine._store_complete_copy_exclusion_proofs = {}
        engine._store._conn.execute('DELETE FROM host_uid_bindings WHERE store_id = ?', (old,))
        engine._store._conn.commit()
        node = SummaryNode(session_id='S', summary='Synthetic new summary', depth=0,
                           source_ids=[new], source_type='messages', token_count=3, source_token_count=100)
        with pytest.raises(LifecyclePublicationConflictError, match='version proof'):
            engine._publish_leaf_node(node, level=1, model='synthetic', expected_frontier=0,
                                      consumed_store_ids=[new], publication_excluded_store_ids=[old],
                                      filter_exclusion_proofs={}, carried_ranges=[])
        assert engine._dag.get_session_nodes('S') == []
        assert engine._lifecycle.get_by_conversation('conv').current_frontier_store_id == 0
    finally:
        engine.shutdown()


@pytest.mark.parametrize('restart', [False, True])
def test_uidless_replay_drains_sparse_gap_exactly_once(tmp_path, monkeypatch, restart):
    import hermes_lcm.engine as module
    from hermes_lcm.tokens import count_messages_tokens
    monkeypatch.setattr(module, 'summarize_with_escalation', lambda **kwargs: ('Synthetic earlier turns.', 1))
    rows = [{'role': 'assistant', 'content': f'synthetic {i} ' + 'word ' * 500} for i in range(6)]
    engine = _engine(tmp_path)
    try:
        ids = [engine._store.append('S', row) for row in rows]
        before = engine._store.get_session_messages('S')
        unknown = {'role': 'assistant', 'content': 'synthetic unresolved head ' + 'other ' * 500}
        view = [unknown, *rows[1:]]
        engine._ingest_cursor = len(view)
        first = engine.compress(view, current_tokens=100000, force=True)
        assert any(unknown['content'].strip() in m.get('content', '') for m in first)
        def sources():
            return [sid for n in engine._dag.get_session_nodes('S')
                    if n.source_type == 'messages' for sid in n.source_ids]
        assert set(sources()) == set(ids[1:4])
        assert engine._last_compacted_store_id == 0
        if restart:
            engine.shutdown()
            engine = _engine(tmp_path)
        # The restored occurrence is now pairable by content; no UID is available.
        engine._ingest_cursor = len(rows)
        result = engine.compress(rows, current_tokens=100000, force=True)
        assert engine._last_compression_status != 'error'
        assert sorted(sources()) == ids[:4]
        # Publication advances through the new leaf's bound; later coverage
        # remains in the sparse source set until a subsequent frontier advance.
        assert engine._last_compacted_store_id == ids[0]
        assert engine._store.get_session_messages('S') == before
        assert count_messages_tokens(result) < count_messages_tokens(rows)
        assert all(any(row['content'].strip() in m.get('content', '') for m in result) for row in rows[-2:])
        engine._ingest_cursor = len(rows)
        engine.compress(rows, current_tokens=100000, force=True)
        assert engine._last_compression_status != 'error'
        assert sorted(sources()) == ids[:4]
        assert engine._store.get_session_messages('S') == before
    finally:
        engine.shutdown()


@pytest.mark.parametrize('role', ['user', 'assistant', 'tool'])
def test_rewrite_rescue_never_claims_only_one_representation(tmp_path, monkeypatch, role):
    import hermes_lcm.engine as module
    from hermes_lcm.tokens import count_messages_tokens
    calls = []

    def summarize(**kwargs):
        calls.append(kwargs['text'])
        if len(calls) == 1:
            raise TimeoutError('synthetic first attempt timeout')
        return 'Synthetic complete occurrence.', 1

    monkeypatch.setattr(module, 'summarize_with_escalation', summarize)
    engine = _engine(tmp_path)
    try:
        original = {'role': role, 'content': 'DURABLE_MARKER ' + 'stored ' * 500, 'message_uid': 'rewritten'}
        host = {**original, 'content': 'REWRITE_MARKER ' + 'rewritten ' * 500}
        prefix = []
        if role == 'tool':
            call = {'role': 'assistant', 'content': '', 'message_uid': 'call', 'tool_calls': [
                {'id': 'synthetic-call', 'type': 'function', 'function': {'name': 'read', 'arguments': '{}'}}]}
            prefix = [call]
            original['tool_call_id'] = host['tool_call_id'] = 'synthetic-call'
        tail = [{'role': 'user', 'content': 'synthetic fresh question', 'message_uid': 'tail-user'},
                {'role': 'assistant', 'content': 'synthetic fresh answer', 'message_uid': 'tail-assistant'}]
        ids = []
        for row in [*prefix, original, *tail]:
            sid = engine._store.append('S', _plain(row))
            bind(engine, row['message_uid'], sid)
            ids.append(sid)
        before = engine._store.get_session_messages('S')
        view = [*prefix, host, *tail]
        engine._ingest_cursor = len(view)
        if role == 'tool':
            # The existing tool-group boundary already makes this indivisible.
            with pytest.raises(TimeoutError, match='synthetic first attempt timeout'):
                engine.compress(view, current_tokens=100000, force=True)
            first = view
        else:
            first = engine.compress(view, current_tokens=100000, force=True)
        # An unsafe prefix retry must neither make a paid call nor claim the source.
        assert len(calls) == 1
        assert engine._dag.get_session_nodes('S') == []
        assert any(host['content'].strip() in m.get('content', '') for m in first)
        assert engine._store.get_session_messages('S') == before
        engine._ingest_cursor = len(view)
        result = engine.compress(view, current_tokens=100000, force=True)
        assert len(calls) == 2
        assert 'DURABLE_MARKER' in calls[-1] and 'REWRITE_MARKER' in calls[-1]
        sources = [sid for n in engine._dag.get_session_nodes('S')
                   if n.source_type == 'messages' for sid in n.source_ids]
        assert sorted(sources) == ids[:-2]
        assert count_messages_tokens(result) < count_messages_tokens(view)
        assert engine._store.get_session_messages('S') == before
        assert all(any(row['content'] in m.get('content', '') for m in result) for row in tail)
    finally:
        engine.shutdown()
