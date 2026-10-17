"""Pinned captures survive dedup, and pinning propagates through duplicate/supersede."""

from __future__ import annotations

import copy
import uuid

import pytest


def _cfg(memohood):
    return copy.deepcopy(memohood.config.DEFAULTS)


def _conn(memohood):
    return memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))


def _insert(conn, memohood, content, *, pinned=False, created_offset=0.0, kind="fact"):
    now = memohood.db.now() + created_offset
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
            (cid, content, kind, int(pinned), now, now, now, now),
        )
    return cid


def _active(conn, cid):
    return conn.execute("SELECT invalidated_at FROM captures WHERE id=?", (cid,)).fetchone()[0] is None


def test_fallback_dedup_keeps_pinned_newer_twin(memohood):
    conn = _conn(memohood)
    older = _insert(conn, memohood, "у меня аллергия на арахис", created_offset=-10)
    newer = _insert(conn, memohood, "у меня аллергия на арахис", pinned=True)
    result = memohood.consolidate.run_dedup(conn, _cfg(memohood))
    assert result["merged"] == 1
    assert _active(conn, newer) and not _active(conn, older)
    conn.close()


def test_fallback_dedup_never_merges_two_pinned(memohood):
    conn = _conn(memohood)
    a = _insert(conn, memohood, "меня зовут Влад", pinned=True, created_offset=-10)
    b = _insert(conn, memohood, "меня зовут Влад", pinned=True)
    assert memohood.consolidate.run_dedup(conn, _cfg(memohood))["merged"] == 0
    assert _active(conn, a) and _active(conn, b)
    conn.close()


def test_vec_dedup_keeps_pinned_twin(memohood):
    conn = _conn(memohood)
    if not memohood.db.ensure_vec_table(conn, 3):
        pytest.skip("sqlite-vec unavailable")
    older = _insert(conn, memohood, "группа крови вторая", created_offset=-10)
    newer = _insert(conn, memohood, "у меня вторая группа крови", pinned=True)
    vec = memohood._engine.embed.serialize_vector([1.0, 0.0, 0.0])
    with conn:
        for cid in (older, newer):
            conn.execute("INSERT INTO captures_vec(capture_id, embedding) VALUES (?, ?)", (cid, vec))
    assert memohood.consolidate.run_dedup(conn, _cfg(memohood))["merged"] == 1
    assert _active(conn, newer) and not _active(conn, older)
    conn.close()


def test_duplicate_with_pinned_pins_existing(memohood, monkeypatch):
    conn = _conn(memohood)
    existing = _insert(conn, memohood, "проект называется Феникс")
    monkeypatch.setattr(
        memohood.capture, "_nearest_captures",
        lambda c, content, cfg, *, k=5: ([{"id": existing, "content": "", "cosine": 0.99}], None),
    )
    result = memohood.capture._store_capture(
        conn, "проект называется Феникс", kind="fact", notability="medium",
        source="EXTRACTED", pinned=True, session_id="s1", cfg=_cfg(memohood),
    )
    assert result["action"] == "duplicate"
    assert conn.execute("SELECT pinned FROM captures WHERE id=?", (existing,)).fetchone()[0] == 1
    conn.close()


def test_supersede_inherits_pinned(memohood, monkeypatch):
    conn = _conn(memohood)
    old = _insert(conn, memohood, "я живу в Москве", pinned=True, kind="persona")
    monkeypatch.setattr(
        memohood.capture, "_nearest_captures",
        lambda c, content, cfg, *, k=5: ([{"id": old, "content": "я живу в Москве", "cosine": 0.93}], None),
    )
    monkeypatch.setattr(
        memohood.extract_llm, "judge",
        lambda *a, **kw: {"action": "supersede", "supersedes_id": old, "reasoning": ""},
    )
    new = memohood.capture._store_capture(
        conn, "я живу в Казани", kind="persona", notability="medium",
        source="EXTRACTED", pinned=False, session_id="s1", cfg=_cfg(memohood),
    )
    assert new["action"] == "supersede"
    assert conn.execute("SELECT pinned FROM captures WHERE id=?", (new["capture_id"],)).fetchone()[0] == 1
    conn.close()
