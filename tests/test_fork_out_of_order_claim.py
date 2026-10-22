"""Fork regression (default 20261020_190551_3754ddd7, sion 20261015_193926_ec05b251, 10-22).

The host replayed newer copies of a tool run before an older user turn, so the host row that owns
the user turn's store row sat behind host rows mapped to LATER store rows. The store-complete leaf
filled its budget with hidden rows first, never reached that host row, the contiguity check found
the user row missing and cut every leaf to nothing, and compaction stopped on every pass
("cannot form a contiguous stored prefix", logged on v0.28.0 as an unresolved retained occurrence).
The leaf now reads that store row from its stored bytes at its store position.
"""
from tests.test_issue_1162_retained_escape import PAD, _engine, _rows
from hermes_lcm.tokens import count_message_tokens


def test_owner_behind_later_rows_does_not_empty_the_leaf(tmp_path):
    engine = _engine(tmp_path, False, fresh_tail_count=1)
    try:
        def row(tag, role="assistant"):
            return {"role": role, "content": f"{tag} " + PAD}
        u1, u2, a2 = row("u1", "user"), row("u2", "user"), row("a2")
        hidden = [row(f"h{i}") for i in range(4)]
        u3 = row("u3", "user")
        engine.ingest([u1, u2, *hidden, a2, u3])
        stored = _rows(engine)
        ids = {r["content"].split(" ", 1)[0]: int(r["store_id"]) for r in stored}
        mapping = {id(a2): ids["a2"], id(u3): ids["u3"]}
        claims = {id(u2): [ids["u2"]]}
        chunk = [a2, u2, u3]  # host order: a2 (a later store row) before the row that owns u2
        budget = 2 * count_message_tokens(hidden[0]) + 1  # hidden rows fill it before a2/u2 are reached
        result = engine._store_complete_input(chunk, claims, mapping, chunk, chunk, ids["u1"], {}, budget,
                                              scanned=([r for r in stored if int(r["store_id"]) > ids["u1"]], False))
        assert result, engine._store_complete_empty_reason
        covered = sorted(s for _m, sids in result for s in sids)
        assert covered[0] == ids["u2"] and covered.count(ids["u2"]) == 1, (covered, ids)
        assert covered == list(range(covered[0], covered[-1] + 1)), covered  # a contiguous store prefix
        assert _rows(engine) == stored  # nothing written
    finally:
        engine.shutdown()
