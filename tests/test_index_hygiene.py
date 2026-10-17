"""Invalidated captures leave captures_fts/captures_vec in the same transaction."""

from __future__ import annotations

import copy
import uuid

import pytest


def _cfg(memohood):
    return copy.deepcopy(memohood.config.DEFAULTS)


def _conn(memohood):
    return memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))


def _fts_ids(conn):
    return {r["capture_id"] for r in conn.execute("SELECT capture_id FROM captures_fts").fetchall()}


def _insert(conn, memohood, content, *, kind="fact", age_days=0.0, pinned=False):
    now = memohood.db.now()
    seen = now - age_days * 86400.0
    cid = uuid.uuid4().hex
    with conn:
        conn.execute(
            """
            INSERT INTO captures(
                id, content, kind, confidence, notability, source, pinned,
                supersedes, history, session_id, message_id, tags, last_seen_at,
                created_at, updated_at, valid_from, invalidated_at, embed_signature
            ) VALUES (?, ?, ?, 1.0, 'medium', 'EXTRACTED', ?, '', '', 's1', NULL, '', ?, ?, ?, ?, NULL, NULL)
            """,
            (cid, content, kind, int(pinned), seen, seen, seen, seen),
        )
        conn.execute(
            "INSERT INTO captures_fts(content, content_stem, capture_id) VALUES (?, ?, ?)",
            (content, memohood._engine.stem.stem_ru(content), cid),
        )
    return cid


def test_supersede_drops_old_row_from_fts(memohood, monkeypatch):
    conn = _conn(memohood)
    old_id = _insert(conn, memohood, "мы используем MySQL", kind="decision")
    monkeypatch.setattr(
        memohood.capture, "_nearest_captures",
        lambda c, content, cfg, *, k=5: ([{"id": old_id, "content": "мы используем MySQL", "cosine": 0.93}], None),
    )
    monkeypatch.setattr(
        memohood.extract_llm, "judge",
        lambda *a, **kw: {"action": "supersede", "supersedes_id": old_id, "reasoning": ""},
    )
    new = memohood.capture._store_capture(
        conn, "мы используем PostgreSQL", kind="decision", notability="medium",
        source="EXTRACTED", pinned=False, session_id="s1", cfg=_cfg(memohood),
    )
    assert new["action"] == "supersede"
    assert old_id not in _fts_ids(conn)
    assert new["capture_id"] in _fts_ids(conn)
    conn.close()


def test_decay_archive_drops_from_fts(memohood):
    conn = _conn(memohood)
    cid = _insert(conn, memohood, "старое событие", kind="event", age_days=400)
    memohood.consolidate.run_decay(conn, _cfg(memohood))
    assert conn.execute("SELECT invalidated_at FROM captures WHERE id=?", (cid,)).fetchone()[0] is not None
    assert cid not in _fts_ids(conn)
    conn.close()


def test_fallback_dedup_drops_from_fts(memohood):
    conn = _conn(memohood)
    a = _insert(conn, memohood, "одинаковый факт")
    b = _insert(conn, memohood, "одинаковый факт")
    memohood.consolidate.run_dedup(conn, _cfg(memohood))
    assert len({a, b} & _fts_ids(conn)) == 1
    conn.close()


def test_drop_from_vec_table(memohood):
    conn = _conn(memohood)
    if not memohood.db.ensure_vec_table(conn, 3):
        pytest.skip("sqlite-vec unavailable")
    cid = _insert(conn, memohood, "факт с вектором")
    with conn:
        conn.execute(
            "INSERT INTO captures_vec(capture_id, embedding) VALUES (?, ?)",
            (cid, memohood._engine.embed.serialize_vector([1.0, 0.0, 0.0])),
        )
        memohood.db.drop_capture_from_indexes(conn, [cid])
    assert conn.execute("SELECT COUNT(*) FROM captures_vec WHERE capture_id=?", (cid,)).fetchone()[0] == 0
    assert cid not in _fts_ids(conn)
    conn.close()


def test_low_vector_top_falls_back_to_fts_duplicate(memohood, monkeypatch):
    # The FTS-only twin (written while the embedder was down) must still be
    # found as a duplicate when the vector KNN only returns an unrelated row.
    conn = _conn(memohood)
    twin = _insert(conn, memohood, "проект называется Феникс и живёт на сервере")
    monkeypatch.setattr(
        memohood.capture, "_nearest_captures",
        lambda c, content, cfg, *, k=5: ([{"id": "other", "content": "другое", "cosine": 0.30}], None),
    )
    result = memohood.capture._store_capture(
        conn, "проект называется Феникс и живёт на сервере", kind="fact", notability="medium",
        source="EXTRACTED", pinned=False, session_id="s1", cfg=_cfg(memohood),
    )
    assert result["action"] == "duplicate"
    assert result["capture_id"] == twin
    conn.close()
