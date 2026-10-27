"""fork: fork_copy_quarantine takes re-stored copies of already stored rows out of selection, losslessly.

Proofs pinned: uid identity (binding or host transcript) AND equal bytes; the earliest row of each uid stays; an
edit (same uid, new bytes) stays; restore() puts every row back; a frontier on a copy moves down."""

from __future__ import annotations

import sqlite3


from hermes_lcm import fork_copy_quarantine as fcq
from hermes_lcm import fork_summary_quarantine as fsq


def _db(tmp_path):
    conn = sqlite3.connect(tmp_path / "lcm.db", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE messages (store_id INTEGER PRIMARY KEY, session_id TEXT, conversation_id TEXT, role TEXT,
            content TEXT, tool_call_id TEXT, tool_calls TEXT, observed_at REAL);
        CREATE TABLE host_uid_bindings (store_id INTEGER, uid TEXT, lineage_key TEXT, kind TEXT);
        CREATE TABLE lcm_lifecycle_state (conversation_id TEXT PRIMARY KEY, current_session_id TEXT,
            current_frontier_store_id INTEGER, last_finalized_session_id TEXT, last_finalized_frontier_store_id INTEGER,
            updated_at REAL);
    """)
    return conn


def _row(conn, sid, role, content, ts, uid=None, tcid=None, bind=True):
    conn.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?)", (sid, "S", "conv", role, content, tcid, None, ts))
    if uid and bind:
        conn.execute("INSERT INTO host_uid_bindings VALUES (?,?,?,?)", (sid, uid, "abcd:S", "canonical"))


def _state(tmp_path, rows):
    st = sqlite3.connect(tmp_path / "state.db")
    st.execute("CREATE TABLE messages (session_id TEXT, message_uid TEXT, timestamp REAL, role TEXT, content TEXT, "
               "tool_call_id TEXT)")
    st.executemany("INSERT INTO messages VALUES ('S',?,?,?,?,?)", rows)
    st.commit()
    st.close()
    return str(tmp_path / "state.db")


def test_copies_quarantined_originals_kept_and_restorable(tmp_path):
    conn = _db(tmp_path)
    _row(conn, 1, "user", "hello", 1.0, "u1")
    _row(conn, 2, "assistant", "hi there", 2.0, "u2")
    _row(conn, 3, "tool", "big output " * 20, 3.0, "u3", "c1", bind=False)
    # restamped re-store at 9.x: u2 bound by uid, u3 known only via the host transcript stamp, u1 edited
    _row(conn, 4, "assistant", "hi there", 9.0, "u2")
    _row(conn, 5, "tool", fcq.PLACEHOLDER, 9.1, "u3", "c1", bind=False)
    _row(conn, 6, "user", "hello again", 9.2, "u1")  # same uid, new bytes: an edit, never a copy
    conn.execute("INSERT INTO lcm_lifecycle_state VALUES ('conv','S',5,'S',2,0)")
    state = _state(tmp_path, [("u3", 3.0, "tool", "big output " * 20, "c1"), ("u3", 9.1, "tool", fcq.PLACEHOLDER, "c1")])
    plan = fcq.plan(conn, "S", state_db=state)
    assert [(s, e) for s, e, _c in plan["rows"]] == [(4, 2), (5, 3)]
    assert plan["frontier_moves"] == [("conv", "current_frontier_store_id", 5, 3)]
    fcq.apply(conn, "S", state_db=state)
    left = [r[0] for r in conn.execute("SELECT store_id FROM messages WHERE session_id='S' ORDER BY store_id")]
    assert left == [1, 2, 3, 6]
    assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 6  # nothing deleted
    assert conn.execute("SELECT current_frontier_store_id FROM lcm_lifecycle_state").fetchone()[0] == 3
    assert fsq.restore(conn, [4, 5]) == 2
    assert [r[0] for r in conn.execute("SELECT store_id FROM messages WHERE session_id='S' ORDER BY store_id")] \
        == [1, 2, 3, 4, 5, 6]


def test_without_uid_proof_equal_bytes_are_not_a_copy(tmp_path):
    conn = _db(tmp_path)
    _row(conn, 1, "assistant", "ok", 1.0, "u1")
    _row(conn, 2, "assistant", "ok", 2.0, "u2")  # a second, genuine "ok" reply: other uid
    _row(conn, 3, "assistant", "ok", 3.0)  # no uid at all
    assert fcq.plan(conn, "S")["rows"] == []


def test_dry_run_cli_does_not_write(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    conn = _db(tmp_path)
    _row(conn, 1, "assistant", "hi", 1.0, "u1")
    _row(conn, 2, "assistant", "hi", 2.0, "u1")
    conn.close()
    script = Path(fcq.__file__).resolve().parent / "scripts" / "fork_quarantine_copy_rows.py"
    out = subprocess.run([sys.executable, str(script), "--db", str(tmp_path / "lcm.db"), "--session", "S"],
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(out)[0]["copies"] == 1
    check = sqlite3.connect(tmp_path / "lcm.db")
    assert check.execute("SELECT count(*) FROM messages WHERE session_id='S'").fetchone()[0] == 2
    refused = subprocess.run([sys.executable, str(script), "--db", str(tmp_path / "lcm.db"), "--session", "S",
                              "--apply"], capture_output=True, text=True)
    assert refused.returncode == 2
