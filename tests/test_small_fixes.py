"""Small hardening fixes: model-load lock, unit vectors, FTS term cap,
rewind cache drop, built-in memory replace/remove mirroring."""

from __future__ import annotations

import copy
import math
import sys
import threading
import time
import types


def _cfg(memohood):
    return copy.deepcopy(memohood.config.DEFAULTS)


def _provider(memohood):
    p = memohood.provider.MemoHoodMemoryProvider()
    p.initialize("s1", hermes_home=str(memohood._hermes_home_for_test), platform="cli")
    return p


def test_local_model_loaded_once_under_concurrency(memohood, monkeypatch):
    created = []

    class FakeTextEmbedding:
        def __init__(self, model_name):
            time.sleep(0.05)
            created.append(model_name)

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding))
    embed = memohood._engine.embed
    threads = [threading.Thread(target=embed._get_local_model, args=("m",)) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert created == ["m"]


def test_embed_texts_returns_unit_vectors(memohood, monkeypatch):
    embed = memohood._engine.embed
    monkeypatch.setattr(embed, "_embed_cloudflare", lambda texts, model, **kw: [[3.0, 4.0, 0.0] for _ in texts])
    cfg = {"embedder": {"provider": "cloudflare", "model": "x", "dims": 3}}
    (vec,) = embed.embed_texts(["a"], cfg)
    assert vec == [0.6, 0.8, 0.0]
    assert math.isclose(sum(x * x for x in vec), 1.0)


def test_fts_match_expression_is_capped(memohood):
    retrieve = memohood._engine.retrieve
    query = " ".join(f"v{i}" for i in range(200))  # 200 distinct coded terms
    expr = retrieve._build_match_expression(query, stem_col="content_stem", raw_col="content")
    assert expr.count(" OR ") + 1 == retrieve.MAX_MATCH_TERMS


def test_rewound_session_switch_drops_cached_prefetch(memohood):
    p = _provider(memohood)
    with p._prefetch_lock:
        p._prefetch_cache["s1"] = ("q", "старый recall", 1)
        p._prefetch_cache["other"] = ("q", "чужой recall", 1)
    p.on_session_switch("s1", rewound=True)
    assert "s1" not in p._prefetch_cache
    assert "other" in p._prefetch_cache
    p.shutdown()


def _active_contents(conn):
    return sorted(r["content"] for r in conn.execute("SELECT content FROM captures WHERE invalidated_at IS NULL"))


def test_memory_write_replace_supersedes_mirror(memohood):
    p = _provider(memohood)
    p.on_memory_write("add", "user", "Пользователь живёт в Москве")
    p._join_background(timeout=5)
    p.on_memory_write(
        "replace", "user", "Пользователь живёт в Казани",
        metadata={"previous_content": "Пользователь живёт в Москве"},
    )
    p._join_background(timeout=5)
    conn = p._connect()
    assert _active_contents(conn) == ["Пользователь живёт в Казани"]
    row = conn.execute("SELECT history, supersedes FROM captures WHERE invalidated_at IS NULL").fetchone()
    assert "Пользователь живёт в Москве" in row["history"] and row["supersedes"]
    conn.close()
    p.shutdown()


def test_memory_write_remove_invalidates_mirror(memohood):
    p = _provider(memohood)
    p.on_memory_write("add", "memory", "Деплой идёт через GitHub Actions")
    p._join_background(timeout=5)
    p.on_memory_write("remove", "memory", "", metadata={"previous_content": "Деплой идёт через GitHub Actions"})
    p._join_background(timeout=5)
    conn = p._connect()
    assert _active_contents(conn) == []
    assert conn.execute("SELECT COUNT(*) FROM captures_fts").fetchone()[0] == 0
    conn.close()
    p.shutdown()
