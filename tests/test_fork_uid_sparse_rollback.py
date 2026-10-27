"""The previous fork reads and writes sparse leaves without changing raw history."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

from tests.test_fork_rewritten_uid_mapping import _engine, _publish, bind
from tests.test_host_uid_shadow import _skip_or_fail_in_ci

BASE = '2133610fddba06cc92a0984012793f5c98da8fa1'
OLD_WRITER = r'''
import importlib.util, json, sys
old, root, db, home = sys.argv[1:]
sys.path.insert(0, root)
spec = importlib.util.spec_from_file_location('hermes_lcm', old + '/__init__.py', submodule_search_locations=[old])
sys.modules['hermes_lcm'] = importlib.util.module_from_spec(spec)
import hermes_lcm.engine as module
from hermes_lcm.config import LCMConfig
module.summarize_with_escalation = lambda **kwargs: ('Synthetic earlier gap.', 1)
engine = module.LCMEngine(config=LCMConfig(database_path=db, fresh_tail_count=1, leaf_chunk_tokens=1,
                                         temporal_rollups_enabled=False), hermes_home=home)
try:
    engine.on_session_start('S', platform='cli', context_length=200000, conversation_id='conv')
    assert engine._last_compacted_store_id == 0
    before = engine._store.get_session_messages('S')
    tail = {'role': 'user', 'content': 'synthetic fresh input', 'message_uid': 'new-tail'}
    view = [{**engine._store.to_openai_msg(row), 'message_uid': f"uid-{row['store_id'] - 1}"}
            for row in before[:2]] + [tail]
    engine.ingest(view)
    engine.compress(view, force=True)
    after = engine._store.get_session_messages('S')
    assert after[:3] == before
    sources = [sid for n in engine._dag.get_session_nodes('S') if n.source_type == 'messages' for sid in n.source_ids]
    assert len(sources) == len(set(sources)) and set(sources) == {1, 2, 3}
    print(json.dumps({'row_count': len(after), 'source_ids': sorted(sources)}))
finally:
    engine.shutdown()
'''


def test_previous_fork_drains_sparse_gap_without_rewriting_history(tmp_path):
    root = Path(__file__).resolve().parents[1]
    if subprocess.run(['git', '-C', str(root), 'cat-file', '-e', f'{BASE}^{{commit}}'], capture_output=True).returncode:
        _skip_or_fail_in_ci('the previous fork commit is required for sparse rollback validation')
    archive = subprocess.run(['git', '-C', str(root), 'archive', BASE], check=True, capture_output=True).stdout
    old = tmp_path / 'old-reader'
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(old, filter='data')
    engine = _engine(tmp_path)
    try:
        for i in range(3):
            sid = engine._store.append('S', {'role': 'assistant', 'content': f'synthetic {i} ' + 'word ' * 100})
            bind(engine, f'uid-{i}', sid)
        _publish(engine, 3, 'uid-2')
        before = engine._store.get_session_messages('S')
    finally:
        engine.shutdown()
    result = subprocess.run([sys.executable, '-B', '-c', OLD_WRITER, str(old), str(root),
                             str(tmp_path / 'lcm.db'), str(tmp_path / 'home')],
                            cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr[-3000:]
    assert json.loads(result.stdout) == {'row_count': 4, 'source_ids': [1, 2, 3]}
    engine = _engine(tmp_path)
    try:
        assert engine._store.get_session_messages('S')[:3] == before
        assert len(engine._store.get_session_messages('S')) == 4
    finally:
        engine.shutdown()
