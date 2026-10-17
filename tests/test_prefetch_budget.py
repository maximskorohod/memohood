"""Prefetch network budget: short timeout, no retries, overall deadline."""

from __future__ import annotations

import copy
import time


def _cfg(memohood):
    return copy.deepcopy(memohood.config.DEFAULTS)


def _conn(memohood):
    return memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))


def _seed(memohood, conn):
    memohood.capture.manual_capture(
        conn, "мы используем PostgreSQL для проекта", kind="decision", session_id="s1", cfg=_cfg(memohood),
    )


def test_net_limits(memohood):
    retrieve = memohood._engine.retrieve
    assert retrieve._net_limits(None, None) == {}
    limits = retrieve._net_limits(time.monotonic() + 10, 3.0)
    assert limits["max_retries"] == 0 and limits["timeout"] == 3.0
    short = retrieve._net_limits(time.monotonic() + 1.0, 3.0)
    assert short["timeout"] <= 1.0
    assert retrieve._net_limits(time.monotonic() - 1, 3.0) is None


def test_hybrid_search_passes_bounded_limits(memohood, monkeypatch):
    conn = _conn(memohood)
    _seed(memohood, conn)
    retrieve = memohood._engine.retrieve
    seen = {}
    monkeypatch.setattr(retrieve, "_vec_ready", lambda c: True)

    def fake_embed(texts, cfg, **kw):
        seen["embed"] = kw
        raise retrieve.embed_mod.EmbedError("offline")

    def fake_rerank(query, cands, cfg, **kw):
        seen["rerank"] = kw
        return cands, "rrf-only"

    monkeypatch.setattr(retrieve.embed_mod, "embed_texts", fake_embed)
    monkeypatch.setattr(retrieve.rerank_mod, "rerank", fake_rerank)
    out = retrieve.hybrid_search(
        conn, "PostgreSQL", 4, _cfg(memohood), deadline=time.monotonic() + 6, net_timeout=3.0,
    )
    assert out and out[0]["text"].startswith("мы используем")
    assert seen["embed"]["max_retries"] == 0 and seen["embed"]["timeout"] <= 3.0
    assert seen["rerank"]["max_retries"] == 0 and seen["rerank"]["timeout"] <= 3.0
    conn.close()


def test_expired_deadline_falls_back_to_fts_only(memohood, monkeypatch):
    conn = _conn(memohood)
    _seed(memohood, conn)
    retrieve = memohood._engine.retrieve
    monkeypatch.setattr(retrieve, "_vec_ready", lambda c: True)

    def boom(*a, **kw):
        raise AssertionError("no network step after the deadline")

    monkeypatch.setattr(retrieve.embed_mod, "embed_texts", boom)
    monkeypatch.setattr(retrieve.rerank_mod, "rerank", boom)
    out = retrieve.hybrid_search(conn, "PostgreSQL", 4, _cfg(memohood), deadline=time.monotonic() - 1, net_timeout=3.0)
    assert out and out[0]["mode"] == "rrf-only" and out[0]["source"] == "fts"
    conn.close()


def test_prefetch_uses_deadline(memohood, monkeypatch):
    p = memohood.provider.MemoHoodMemoryProvider()
    p.initialize("s1", hermes_home=str(memohood._hermes_home_for_test), platform="cli")
    seen = {}

    def spy(conn, query, k, cfg, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(memohood._engine.retrieve, "hybrid_search", spy)
    p.prefetch("какой у нас стек базы данных проекта")
    assert seen["net_timeout"] == memohood.provider._PREFETCH_NET_TIMEOUT_S
    assert 0 < seen["deadline"] - time.monotonic() <= memohood.provider._PREFETCH_DEADLINE_S
    p.shutdown()
