"""Preserved-objective nesting and pile-up across compactions (synthetic host replay).

Host model, per compaction: the host persists the list ``compress`` returned with fresh stamps,
its process restarts (cold resume, a new engine on the same store), and the user's next message
lands right after LCM's objective row, which the host merges into it (Hermes repairs role
alternation as ``prev + "\n\n" + next``). Before the fix each compaction stored that merged
row whole (objective + summary prefix + the user's line) as an unpaired raw user row, re-emitted
it as the next objective (nesting the summary prefix), and the frontier stopped moving.
"""

import itertools

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.reconcile import _PRESERVED_OBJECTIVE_CONTEXT_PREFIX as OBJ
from hermes_lcm.tokens import count_messages_tokens

COMPACTIONS = 8
SUMMARY_HEADER = "Summary (d"


def _engine(tmp_path, **config):
    cfg = LCMConfig(database_path=str(tmp_path / "lcm.db"), fresh_tail_count=8, **config)
    engine = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    engine.on_session_start("objective-pileup", platform="cli", context_length=200_000)
    return engine


def _host_merge(rows):
    out = []
    for row in rows:
        if out and out[-1].get("role") == "user" and row.get("role") == "user":
            out[-1] = {**out[-1], "content": f"{out[-1]['content']}\n\n{row['content']}"}
        else:
            out.append(dict(row))
    return out


def _objective_rows(engine):
    rows = engine._store.get_session_messages(engine.current_session_id)
    return [row for row in rows if str(row.get("content") or "").startswith(OBJ)]


@pytest.fixture
def stub_summary(monkeypatch):
    calls = itertools.count(1)
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **kw: (
        f"Summary {next(calls)} of older work. " + "s" * 3000 + "\nExpand for details about: older work", 1))


def _run_compactions(tmp_path, **config):
    """Drive COMPACTIONS compactions through the host model above; return per-compaction observations."""
    clock = itertools.count(1_700_000_000)
    turn = itertools.count()

    def tool_turns(n):
        rows = []
        for _ in range(n):
            i = next(turn)
            rows += [{"role": "assistant", "content": f"step {i} " + "x" * 200, "timestamp": float(next(clock)),
                      "tool_calls": [{"id": f"c{i}", "type": "function",
                                      "function": {"name": "terminal", "arguments": "{}"}}]},
                     {"role": "tool", "tool_call_id": f"c{i}", "content": f"ok {i} " + "data " * 400,
                      "timestamp": float(next(clock))}]
        return rows

    engine = _engine(tmp_path, **config)
    threshold = engine.threshold_tokens
    assert threshold > 0
    view = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "OBJECTIVE: build the dashboard. " + "detail " * 3000,
             "timestamp": float(next(clock))}, *tool_turns(40)]
    notes, frontiers, live, copies, nested = [], [], [], [], []
    try:
        for c in range(COMPACTIONS):
            result = engine.compress(view)
            frontiers.append(engine._last_compacted_store_id)
            live.append(count_messages_tokens(result))
            head = next((m for m in result if m.get("role") == "user"), {})
            content = str(head.get("content") or "")
            copies.append(len(_objective_rows(engine)))
            objective = content.split("\n\n---\n\n", 1)[0] if content.startswith(OBJ) else ""
            nested.append((SUMMARY_HEADER in objective, len(objective)))
            engine.shutdown()
            engine = _engine(tmp_path, **config)
            view = [dict(row, timestamp=float(next(clock))) if row.get("role") != "system" else dict(row)
                    for row in result]
            note = {"role": "user", "content": f"Quick note {c}: keep going and also check item {c}.",
                    "timestamp": float(next(clock))}
            notes.append(note["content"])
            if str(view[1].get("content") or "").startswith(OBJ):
                view = _host_merge([view[0], view[1], note, *view[2:]])
            else:
                view.append(note)
            view += tool_turns(40)
        engine.compress(view)
        stored_user = [str(row["content"]) for row in engine._store.get_session_messages(engine.current_session_id)
                       if row["role"] == "user"]
    finally:
        engine.shutdown()
    return dict(frontiers=frontiers, live=live, threshold=threshold, copies=copies, nested=nested,
                notes=notes, stored_user=stored_user)


def _assert_frontier_moves_and_view_stays_small(run):
    # Every compaction after the first had compactable rows: the frontier moves each time.
    frontiers = run["frontiers"]
    assert all(later > earlier for earlier, later in zip(frontiers[1:], frontiers[2:])), frontiers
    # The live view after each compaction stays far below the threshold.
    assert max(run["live"][1:]) <= 0.3 * run["threshold"], (run["live"], run["threshold"])
    # Lossless: every user line the host merged is stored.
    for text in run["notes"]:
        assert any(text in row for row in run["stored_user"]), text


def test_default_objective_rows_never_stall_the_frontier(tmp_path, stub_summary):
    """Cap off (default, upstream objective contract): merged objective rows are stored whole, but they are
    LCM scaffolding, never unpaired source rows, so they cannot hold the frontier."""
    _assert_frontier_moves_and_view_stays_small(_run_compactions(tmp_path))


def test_capped_objective_does_not_nest_or_pile_up(tmp_path, stub_summary):
    run = _run_compactions(tmp_path, preserved_objective_max_chars=2000)
    _assert_frontier_moves_and_view_stays_small(run)
    assert max(run["copies"]) <= 1, run["copies"]  # at most one stored objective copy, ever
    for index, (has_summary, length) in enumerate(run["nested"]):
        assert not has_summary, f"compaction {index + 1} nested a summary into the objective"
        assert length <= len(OBJ) + 1 + 2000, (index, length)
    for text in run["notes"]:
        assert sum(text in row for row in run["stored_user"]) == 1, text  # each merged line stored once


def test_capped_objective_keeps_user_words_and_drops_only_verified_summaries(tmp_path):
    from hermes_lcm.dag import SummaryNode

    engine = _engine(tmp_path, preserved_objective_max_chars=500)
    try:
        node_id = engine._dag.add_node(SummaryNode(
            session_id=engine.current_session_id, depth=0, summary="Older history.", token_count=3,
            source_token_count=10, source_ids=[], source_type="messages", expand_hint="older", created_at=1.0))
        verified = f"[Recent Summary (d0, node {node_id})]\nOlder history.\n[Expand for details: older]"
        quoted = "[Recent Summary (d0, node 999)]\nquoted by the user"
        source = (f"{OBJ}\nShip the report.\n\n---\n\n{quoted}\n\n---\n\n{verified}\n\nAlso fix the chart. "
                  + "w" * 900)
        part = engine._build_preserved_objective_summary_part({"role": "user", "content": source})
        uncapped = _engine(tmp_path / "off")
        try:
            assert uncapped._build_preserved_objective_summary_part({"role": "user", "content": "x " * 3000}) == (
                f"{OBJ}\n" + "x " * 3000)  # default: byte for byte, no cap
        finally:
            uncapped.shutdown()
    finally:
        engine.shutdown()
    words = part[len(OBJ) + 1:]
    assert words.startswith("Ship the report.") and "quoted by the user" in words and "Also fix the chart." in words
    assert "Older history." not in part and OBJ not in words
    assert len(words) <= 500 and words.endswith(lcm_engine._OBJECTIVE_TRUNCATION_MARK)


def test_repair_script_dry_run_then_apply_supersedes_older_identical_copies(tmp_path, capsys):
    import importlib.util
    import sqlite3
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "repair_preserved_objective_copies",
        Path(__file__).resolve().parents[1] / "scripts" / "repair_preserved_objective_copies.py")
    repair = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repair)
    engine = _engine(tmp_path)
    copy = f"{OBJ}\nstale objective"
    session = engine.current_session_id
    engine._store.append_batch(session, [{"role": "user", "content": copy}, {"role": "user", "content": copy},
                                         {"role": "user", "content": copy + " with a merged line"},
                                         {"role": "user", "content": copy}])
    ids = [int(row["store_id"]) for row in _objective_rows(engine)]
    db = Path(engine._store.db_path)
    engine.shutdown()

    def snapshot():
        with sqlite3.connect(db) as conn:
            return conn.execute("SELECT store_id, role, content FROM messages ORDER BY store_id").fetchall()

    before = snapshot()
    assert repair.main(["--db", str(db)]) == 0
    dry = capsys.readouterr().out
    assert '"would_supersede"' in dry and "dry run: 2 row(s) would be marked" in dry
    assert snapshot() == before  # a dry run writes nothing
    with pytest.raises(SystemExit):
        repair.main(["--db", str(db), "--apply"])  # no backup path: refused before any write
    capsys.readouterr()
    backup = tmp_path / "before-repair.db"
    assert repair.main(["--db", str(db), "--apply", "--backup", str(backup)]) == 0
    assert "applied: 2 row(s) marked" in capsys.readouterr().out
    with sqlite3.connect(backup) as conn:
        assert conn.execute("SELECT store_id, role, content FROM messages ORDER BY store_id").fetchall() == before
    assert snapshot() == before  # nothing deleted, no bytes changed
    with sqlite3.connect(db) as conn:
        live = [r[0] for r in conn.execute("SELECT store_id FROM messages WHERE session_id = ? AND content LIKE ? "
                                           "ORDER BY store_id", (session, OBJ + "%"))]
        marked = dict(conn.execute("SELECT store_id, node_ids FROM lcm_fork_quarantined_messages"))
    assert live == [ids[2], ids[3]]  # the newest identical copy and the distinct copy stay live
    assert marked == {ids[0]: f"superseded-by:{ids[3]}", ids[1]: f"superseded-by:{ids[3]}"}
    assert repair.main(["--db", str(db), "--restore", f"{ids[0]},{ids[1]}"]) == 0
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM messages WHERE session_id = ?", (session,)).fetchone()[0] == len(before)
