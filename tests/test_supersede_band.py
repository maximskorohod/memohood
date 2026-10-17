"""Supersede judge for corrections/preferences/decisions below the 0.92 band."""

from __future__ import annotations

import copy

import pytest


def _cfg(memohood):
    return copy.deepcopy(memohood.config.DEFAULTS)


def _setup(memohood, monkeypatch, cosine):
    conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
    old = memohood.capture.manual_capture(
        conn, "мы используем MySQL", kind="decision", session_id="s1", cfg=_cfg(memohood),
    )["capture_id"]
    monkeypatch.setattr(
        memohood.capture, "_nearest_captures",
        lambda c, content, cfg, *, k=5: ([{"id": old, "content": "мы используем MySQL", "cosine": cosine}], None),
    )
    calls = []

    def fake_judge(new_content, candidates, **kw):
        calls.append(candidates)
        return {"action": "supersede", "supersedes_id": old, "reasoning": ""}

    monkeypatch.setattr(memohood.extract_llm, "judge", fake_judge)
    return conn, old, calls


def test_decision_at_0_6_is_judged_when_key_present(memohood, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    conn, old, calls = _setup(memohood, monkeypatch, 0.6)
    result = memohood.capture._store_capture(
        conn, "мы перешли с MySQL на PostgreSQL", kind="decision", notability="medium",
        source="EXTRACTED", pinned=False, session_id="s1", cfg=_cfg(memohood),
    )
    assert len(calls) == 1
    assert result["action"] == "supersede" and result["supersedes"] == old
    conn.close()


def test_no_judge_without_gemini_key(memohood, monkeypatch):
    conn, _old, calls = _setup(memohood, monkeypatch, 0.6)
    result = memohood.capture._store_capture(
        conn, "мы перешли с MySQL на PostgreSQL", kind="decision", notability="medium",
        source="EXTRACTED", pinned=False, session_id="s1", cfg=_cfg(memohood),
    )
    assert calls == [] and result["action"] == "independent"
    conn.close()


@pytest.mark.parametrize("kind,cosine", [("fact", 0.6), ("decision", 0.4)])
def test_no_judge_for_plain_facts_or_low_similarity(memohood, monkeypatch, kind, cosine):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    conn, _old, calls = _setup(memohood, monkeypatch, cosine)
    memohood.capture._store_capture(
        conn, "совсем другой факт про проект", kind=kind, notability="medium",
        source="EXTRACTED", pinned=False, session_id="s1", cfg=_cfg(memohood),
    )
    assert calls == []
    conn.close()
