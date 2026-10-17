"""MemoHoodMemoryProvider lifecycle: initialize, prefetch (+recall text +
reinforce), child/delegated session hard no-op, sync_turn (background
thread), on_pre_compress rescue."""

from __future__ import annotations

import time

import pytest


def _conn(p):
    """Test-side connection for a provider (the provider itself keeps none since hermes 0.22)."""
    c = p.__dict__.get("_test_conn")
    if c is None:
        c = p._connect()
        p.__dict__["_test_conn"] = c
    return c


def _make_provider(memohood, *, agent_context=None, parent_session_id=""):
    p = memohood.provider.MemoHoodMemoryProvider()
    kwargs = {"hermes_home": str(memohood._hermes_home_for_test), "platform": "cli"}
    if agent_context is not None:
        kwargs["agent_context"] = agent_context
    if parent_session_id:
        kwargs["parent_session_id"] = parent_session_id
    p.initialize("s1", **kwargs)
    return p


class TestInitialize:
    def test_initialize_opens_db_and_runs_catch_up(self, memohood):
        p = _make_provider(memohood)
        assert p._ready
        assert (memohood._hermes_home_for_test / "memory.db").exists()
        p.shutdown()

    def test_missing_hermes_home_does_not_crash(self, memohood):
        p = memohood.provider.MemoHoodMemoryProvider()
        p.initialize("s1", platform="cli")  # no hermes_home kwarg at all
        assert not p._ready
        # Every method must degrade gracefully, never raise.
        assert p.prefetch("что угодно") == ""
        assert p.system_prompt_block() == ""
        assert "не инициализирована" in p.handle_tool_call("memohood_stats", {})


class TestHermes022Compat:
    def test_tool_schemas_available_before_initialize(self, memohood):
        # hermes 0.22 snapshots get_tool_schemas() in add_provider(), BEFORE initialize().
        p = memohood.provider.MemoHoodMemoryProvider()
        names = {s["name"] for s in p.get_tool_schemas()}
        assert "memohood_stats" in names

    def test_prefetch_and_tools_work_from_other_threads(self, memohood):
        # hermes 0.22 runs prefetch in spawn_context_thread and tools in executor workers.
        import threading

        p = _make_provider(memohood)
        memohood.capture.manual_capture(
            _conn(p), "Мы подписали договор с новым поставщиком оборудования",
            kind="decision", notability="high", pinned=False, session_id="s1", cfg=p._cfg,
        )
        out = {}

        def worker():
            out["prefetch"] = p.prefetch("расскажи про договора с поставщиком", session_id="s1")
            out["tool"] = p.handle_tool_call("memohood_stats", {})

        t = threading.Thread(target=worker)
        t.start()
        t.join(10)
        assert "договор" in out["prefetch"].lower()
        assert "не удалось" not in out["tool"] and "не инициализирована" not in out["tool"]
        p.shutdown()


class TestChildSessionHardNoOp:
    @pytest.mark.parametrize("agent_context", ["subagent", "cron", "flush"])
    def test_agent_context_child_is_hard_noop(self, memohood, agent_context):
        p = _make_provider(memohood, agent_context=agent_context)
        assert p._is_child is True
        assert p.prefetch("что угодно") == ""
        assert p.system_prompt_block() == ""
        assert p.get_tool_schemas() == []
        p.shutdown()

    def test_parent_session_id_alone_marks_child(self, memohood):
        p = _make_provider(memohood, parent_session_id="parent-123")
        assert p._is_child is True
        assert p.prefetch("что угодно") == ""
        p.shutdown()

    def test_primary_context_is_not_child(self, memohood):
        p = _make_provider(memohood, agent_context="primary")
        assert p._is_child is False
        p.shutdown()


class TestPrefetchRecall:
    def test_prefetch_returns_recall_text_and_reinforces(self, memohood):
        p = _make_provider(memohood)
        result = memohood.capture.manual_capture(
            _conn(p), "Мы подписали договор с новым поставщиком оборудования",
            kind="decision", notability="high", pinned=False, session_id="s1", cfg=p._cfg,
        )
        cid = result["capture_id"]
        before = _conn(p).execute("SELECT last_seen_at FROM captures WHERE id=?", (cid,)).fetchone()["last_seen_at"]

        time.sleep(0.01)
        text = p.prefetch("расскажи про договора с поставщиком", session_id="s1")
        assert "договор" in text.lower()

        after = _conn(p).execute("SELECT last_seen_at FROM captures WHERE id=?", (cid,)).fetchone()["last_seen_at"]
        assert after >= before
        p.shutdown()

    def test_prefetch_empty_query_returns_empty(self, memohood):
        p = _make_provider(memohood)
        assert p.prefetch("") == ""
        assert p.prefetch("   ") == ""
        p.shutdown()

    def test_prefetch_no_hits_returns_empty(self, memohood):
        p = _make_provider(memohood)
        assert p.prefetch("совершенно несвязанный запрос без совпадений") == ""
        p.shutdown()


class TestSyncTurn:
    def test_sync_turn_is_background_and_persists(self, memohood):
        p = _make_provider(memohood)
        p.sync_turn(
            "запомни, что мой любимый редактор - VS Code",
            "хорошо, запомнил",
            session_id="s1",
        )
        p._join_background(timeout=5.0)
        row = _conn(p).execute(
            "SELECT COUNT(*) AS n FROM captures WHERE invalidated_at IS NULL"
        ).fetchone()
        assert row["n"] >= 1
        p.shutdown()

    def test_sync_turn_noop_for_child_session(self, memohood):
        p = _make_provider(memohood, agent_context="subagent")
        p.sync_turn("запомни это навсегда", "ок", session_id="s1")
        p._join_background(timeout=2.0)
        # initialize() still opens a connection for a child context (only
        # catch_up_from_state is skipped) -- sync_turn's own is_child guard
        # is what must prevent any capture from being written.
        assert p._ready
        row = _conn(p).execute(
            "SELECT COUNT(*) AS n FROM captures WHERE invalidated_at IS NULL"
        ).fetchone()
        assert row["n"] == 0
        p.shutdown()

    def test_sync_turn_respects_auto_capture_false(self, memohood):
        p = _make_provider(memohood)
        p._cfg["auto_capture"] = False
        p.sync_turn("запомни, что мой любимый редактор - VS Code", "ок", session_id="s1")
        p._join_background(timeout=2.0)
        row = _conn(p).execute(
            "SELECT COUNT(*) AS n FROM captures WHERE invalidated_at IS NULL"
        ).fetchone()
        assert row["n"] == 0
        p.shutdown()


class TestOnPreCompress:
    def test_rescues_definite_keep_insight_without_llm_call(self, memohood, monkeypatch):
        p = _make_provider(memohood)
        monkeypatch.setattr(
            memohood.extract_llm, "extract",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no LLM call during compression")),
        )
        messages = [
            {"role": "user", "content": "просто болтовня ни о чём"},
            {"role": "user", "content": "Запомни навсегда: имя проекта - Феникс"},
        ]
        summary_note = p.on_pre_compress(messages)
        assert "Феникс" in summary_note or "rescued" in summary_note.lower()
        row = _conn(p).execute(
            "SELECT COUNT(*) AS n FROM captures WHERE invalidated_at IS NULL"
        ).fetchone()
        assert row["n"] == 1
        p.shutdown()

    def test_empty_messages_returns_empty_string(self, memohood):
        p = _make_provider(memohood)
        assert p.on_pre_compress([]) == ""
        p.shutdown()


class TestToolSchemas:
    def test_get_tool_schemas_returns_all_when_primary(self, memohood):
        p = _make_provider(memohood)
        schemas = p.get_tool_schemas()
        names = {s["name"] for s in schemas}
        assert {"memohood_search", "memohood_fetch", "memohood_recall", "memohood_stats", "memohood_capture"} <= names
        p.shutdown()

    def test_handle_tool_call_dispatches(self, memohood):
        p = _make_provider(memohood)
        result = p.handle_tool_call("memohood_capture", {"content": "тестовый факт для инструмента"}, session_id="s1")
        assert "Сохранено" in result or "id=" in result
        p.shutdown()


class TestRecallHygiene:
    def test_migration_v2_drops_tool_rows_from_history_index(self, memohood):
        p = _make_provider(memohood)
        c = _conn(p)
        with c:
            c.execute("DELETE FROM _meta WHERE key='schema_version'")
            c.execute("INSERT INTO _meta(key, value) VALUES ('schema_version', '1')")
            for role in ("user", "assistant", "tool"):
                c.execute(
                    "INSERT INTO messages_fts(content, content_stem, message_id, session_id, role, timestamp) "
                    "VALUES (?, ?, ?, 's1', ?, 0)",
                    (f"текст {role}", f"текст {role}", hash(role) % 10000, role),
                )
        memohood.db.migrate(c)
        roles = sorted(r["role"] for r in c.execute("SELECT role FROM messages_fts"))
        assert roles == ["assistant", "user"]
        assert memohood.db.get_schema_version(c) == 2
        p.shutdown()

    def test_capture_assistant_false_skips_agent_side(self, memohood):
        p = _make_provider(memohood)
        cfg = dict(p._cfg, capture_assistant=False)
        res = memohood.capture.process_turn(
            _conn(p), "привет", "Запомни навсегда: имя проекта - Феникс", session_id="s1", cfg=cfg,
        )
        assert res["assistant"] is None
        n = _conn(p).execute("SELECT COUNT(*) AS n FROM captures WHERE invalidated_at IS NULL").fetchone()["n"]
        assert n == 0
        p.shutdown()


class TestAuditFixes:
    def test_prefetch_never_serves_previous_query(self, memohood):
        p = _make_provider(memohood)
        memohood.capture.manual_capture(
            _conn(p), "Мы подписали договор с новым поставщиком оборудования",
            kind="decision", notability="high", pinned=False, session_id="s1", cfg=p._cfg,
        )
        p.queue_prefetch("расскажи про договора с поставщиком", session_id="s1")
        p._join_background(timeout=5.0)
        assert p.prefetch("совершенно несвязанный запрос без совпадений", session_id="s1") == ""
        p.shutdown()

    def test_recall_status_reflects_last_prefetch(self, memohood):
        p = _make_provider(memohood)
        memohood.capture.manual_capture(
            _conn(p), "Мы подписали договор с новым поставщиком оборудования",
            kind="decision", notability="high", pinned=False, session_id="s1", cfg=p._cfg,
        )
        assert p.prefetch("расскажи про договора с поставщиком", session_id="s1")
        st = p.recall_status()
        assert st is not None and st.count >= 1
        assert p.prefetch("совершенно несвязанный запрос без совпадений", session_id="s1") == ""
        assert p.recall_status() is None
        p.shutdown()

    def test_pre_compress_rescues_are_bounded(self, memohood, monkeypatch):
        p = _make_provider(memohood)
        calls = []
        orig = memohood.capture.extract_and_store
        monkeypatch.setattr(memohood.capture, "extract_and_store", lambda *a, **kw: calls.append(1) or orig(*a, **kw))
        msgs = [{"role": "user", "content": f"Запомни навсегда: факт номер {i} про проект"} for i in range(20)]
        p.on_pre_compress(msgs)
        assert len(calls) <= 5
        p.shutdown()
