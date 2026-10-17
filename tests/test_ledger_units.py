"""Spend ledger: per-token pricing units, query-embedding accounting, gemini ceiling."""

from __future__ import annotations

import copy

import pytest


def _cfg(memohood):
    return copy.deepcopy(memohood.config.DEFAULTS)


def _conn(memohood):
    return memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))


def test_cloudflare_embed_price_is_per_token(memohood):
    est = memohood._engine.ledger.estimate_cost_usd("cloudflare", "embed", 1000)
    assert est == pytest.approx(0.000012, rel=1e-6)


def test_capture_embed_cost_is_not_inflated(memohood, monkeypatch):
    # 1000 captures of ~400 chars (~100 tokens each) must cost ~$0.0012,
    # nowhere near the $5 default ceiling.
    conn = _conn(memohood)
    for _ in range(1000):
        memohood._engine.ledger.record_embed_call(conn, _cfg(memohood), ["x" * 400])
    spent = memohood.db.monthly_spend(conn, "cloudflare")
    assert spent == pytest.approx(0.0012, rel=1e-3)
    conn.close()


def test_migration_v3_recomputes_inflated_cloudflare_spend(memohood):
    conn = _conn(memohood)
    with conn:
        conn.execute("UPDATE _meta SET value = '2' WHERE key = 'schema_version'")
    # Old code: 100k tokens recorded at the per-1K price -> $1.20 instead of $0.0012.
    memohood.db.record_spend(conn, provider="cloudflare", op="embed", units=100_000, est_usd=1.2)
    memohood.db.record_spend(conn, provider="cohere", op="rerank", units=1, est_usd=0.002)
    memohood.db.migrate(conn)
    assert memohood.db.monthly_spend(conn, "cloudflare") == pytest.approx(0.0012, rel=1e-6)
    assert memohood.db.monthly_spend(conn, "cohere") == pytest.approx(0.002)
    conn.close()


def test_local_embedder_not_recorded(memohood):
    conn = _conn(memohood)
    cfg = _cfg(memohood)
    cfg["embedder"]["provider"] = "local"
    memohood._engine.ledger.record_embed_call(conn, cfg, ["text"])
    assert conn.execute("SELECT COUNT(*) AS n FROM spend").fetchone()["n"] == 0
    conn.close()


def test_query_embedding_is_recorded(memohood, monkeypatch):
    conn = _conn(memohood)
    cfg = _cfg(memohood)
    retrieve = memohood._engine.retrieve
    monkeypatch.setattr(retrieve, "_vec_ready", lambda c: True)
    monkeypatch.setattr(retrieve.embed_mod, "embed_texts", lambda texts, cfg, **kw: [[0.0] * 1024])
    retrieve._vector_search(conn, "какой у нас стек", 5, cfg)
    rows = conn.execute("SELECT provider, op, units FROM spend").fetchall()
    assert [(r["provider"], r["op"]) for r in rows] == [("cloudflare", "embed")]
    conn.close()


def test_query_embedding_skipped_over_ceiling(memohood, monkeypatch):
    conn = _conn(memohood)
    cfg = _cfg(memohood)
    cfg["monthly_ceiling_usd"]["cloudflare"] = 0.001
    memohood.db.record_spend(conn, provider="cloudflare", op="embed", units=1, est_usd=0.01)
    retrieve = memohood._engine.retrieve
    monkeypatch.setattr(retrieve, "_vec_ready", lambda c: True)

    def boom(*a, **kw):
        raise AssertionError("must not embed over the ceiling")

    monkeypatch.setattr(retrieve.embed_mod, "embed_texts", boom)
    assert retrieve._vector_search(conn, "запрос", 5, cfg) == []
    conn.close()


def test_gemini_cost_estimate_is_nonzero(memohood):
    units, est = memohood.extract_llm._estimate_gemini_cost(
        {"prompt_tokens": 1000, "completion_tokens": 100}, "sys", "user", "{}",
    )
    assert units == 1100
    assert est == pytest.approx(1000 * 0.10e-6 + 100 * 0.40e-6, rel=1e-6)
    _units, est2 = memohood.extract_llm._estimate_gemini_cost({}, "s" * 400, "u" * 400, "r" * 40)
    assert est2 > 0


def test_borderline_extract_skipped_over_gemini_ceiling(memohood, monkeypatch):
    conn = _conn(memohood)
    cfg = _cfg(memohood)
    cfg["monthly_ceiling_usd"]["gemini"] = 0.001
    memohood.db.record_spend(conn, provider="gemini", op="extract", units=1, est_usd=0.01)

    def boom(*a, **kw):
        raise AssertionError("extract must not be called over the gemini ceiling")

    monkeypatch.setattr(memohood.extract_llm, "extract", boom)
    result = memohood.capture.extract_and_store(
        conn, "я предпочитаю тёмную тему интерфейса", side="user", session_id="s1", cfg=cfg,
    )
    assert result is None
    conn.close()
