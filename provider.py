"""``MemoHoodMemoryProvider`` -- hermes ``MemoryProvider`` implementation for
memohood's dialogue memory (DESIGN_v1.md "Provider ABC methods (implement
against REAL agent/memory_provider.py)").

Every abstract method of ``agent.memory_provider.MemoryProvider`` (v0.18.0)
is implemented; every documented optional hook is overridden too, per
DESIGN_v1.md's method list.

Threading contract (non-negotiable: "sync_turn MUST be non-blocking"):
hermes >= 0.22 calls provider methods from different threads (``prefetch``
runs in ``spawn_context_thread``, tool calls in executor workers, every CLI
turn in a fresh thread), so this provider keeps NO long-lived connection.
Every method that touches memory.db -- synchronous (``prefetch``,
``on_pre_compress``, ``handle_tool_call``) or background (``queue_prefetch``,
``sync_turn``, ``on_memory_write``, ``on_delegation``) -- opens its OWN
short-lived ``sqlite3.Connection`` via :meth:`_connect` and closes it when
done. ``check_same_thread`` stays at sqlite3's safe default. Background
threads are started through hermes' ``spawn_context_thread`` so they inherit
the caller's profile scope (HERMES_HOME override + secret scope).

Child/delegated-session handling (DESIGN_v1.md: "Skip prefetch for
delegated/child sessions ... no memory bleed into KB sub-agents"): hermes-
core already skips ``MemoryProvider.initialize()`` entirely for delegated
subagents via ``delegate_task``'s ``skip_memory=True`` (confirmed by code
inspection of ``tools/delegate_tool.py``) -- so in the common case this
provider is simply never instantiated for a child session at all. This
class still checks ``kwargs["agent_context"]``/``kwargs["parent_session_id"]``
at ``initialize()`` time and sets ``self._is_child`` accordingly, as
defense in depth for any future caller path that initializes a provider for
a non-primary context without ``skip_memory``. When ``self._is_child`` is
set, every recall/capture/tool-schema method is a hard no-op.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

try:  # hermes >= 0.22: background jobs must inherit the caller's contextvars (profile/secret scope)
    from agent.memory_provider import spawn_context_thread as _spawn_context_thread
except ImportError:  # older hermes: plain thread, no profile scope to inherit
    _spawn_context_thread = None

try:  # hermes >= 0.22: deterministic "🧠 MemoHood — recalled N memories" indicator
    from agent.memory_provider import RecallStatus as _RecallStatus
except ImportError:
    _RecallStatus = None

# hermes >= 0.22 already runs sync_turn on its own single background worker,
# which also serializes a provider's writes (turn N before N+1). Spawning another
# thread there would let two turns capture concurrently (duplicate facts, broken
# SUPERSEDE chains). Older hermes called sync_turn on the turn thread.
_HOST_RUNS_SYNC_IN_BACKGROUND = _spawn_context_thread is not None

# on_pre_compress runs on the compression path: bound the per-message network work.
_MAX_PRECOMPRESS_RESCUES = 5

# hermes waits ~8s for prefetch. Each network step on the recall path (query
# embedding, rerank) gets a short timeout and no retries, and the whole recall
# a deadline after which the remaining network steps are skipped (FTS-only).
_PREFETCH_NET_TIMEOUT_S = 3.0
_PREFETCH_DEADLINE_S = 6.0

from . import capture as capture_mod
from . import config as memohood_config
from . import db
from . import gate as gate_mod
from . import graph_rerank as graph_rerank_mod
from . import post_recall
from . import query_norm
from ._engine import retrieve as retrieve_mod

logger = logging.getLogger("memohood.provider")

_CHILD_AGENT_CONTEXTS = frozenset({"subagent", "cron", "flush"})

_SYSTEM_PROMPT_BLOCK = (
    "# Память (memohood)\n"
    "У тебя есть постоянная память диалогов: перед каждым ходом релевантные факты и решения "
    "из прошлых разговоров автоматически подмешиваются в контекст (блок <memory-context>, если "
    "он есть) -- специально вызывать recall не нужно. Явно попросить что-то запомнить навсегда "
    "можно фразой «запомни, что ...» -- такие факты не забываются со временем.\n\n"
    "# Memory (memohood)\n"
    "You have persistent dialogue memory: relevant facts/decisions from past conversations are "
    "automatically recalled before each turn (see the <memory-context> block, if present) -- no "
    "explicit recall call needed. Say \"remember that ...\" to pin a fact permanently."
)


class MemoHoodMemoryProvider(MemoryProvider):
    """Dialogue-memory provider: auto-recall + auto-capture, hybrid
    FTS5(RU-stem)+vector(Cloudflare BGE-M3)+RRF+Cohere-rerank search over
    ``captures`` and a catch-up FTS index of ``state.db`` messages."""

    def __init__(self) -> None:
        self._hermes_home: Optional[str] = None
        self._session_id: str = ""
        self._ready = False
        self._cfg: Dict[str, Any] = {}
        self._is_child = False
        self._agent_context = "primary"

        # session key -> (query, text) of the LAST inline recall; served only for
        # the same query (hermes may re-ask within a turn), never for the next one.
        self._prefetch_cache: Dict[str, tuple] = {}
        self._last_recall_count = 0
        self._prefetch_lock = threading.Lock()
        self._bg_threads: List[threading.Thread] = []
        self._bg_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "memohood"

    def is_available(self) -> bool:
        # Local-only check, no network: sqlite3/stdlib is always available.
        # Optional deps (sqlite-vec/PyStemmer/requests) degrade gracefully
        # inside db.py/_engine -- memohood works FTS-only without them.
        return True

    # -- lifecycle ------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._hermes_home = kwargs.get("hermes_home")
        self._session_id = session_id
        self._agent_context = kwargs.get("agent_context") or "primary"
        parent_session_id = kwargs.get("parent_session_id") or ""
        self._is_child = bool(parent_session_id) or self._agent_context in _CHILD_AGENT_CONTEXTS

        if not self._hermes_home:
            logger.error("memohood.initialize: no hermes_home in kwargs; provider will not persist correctly")
            return

        try:
            self._cfg = memohood_config.get_memohood_config_readonly()
        except Exception:
            logger.warning("memohood.initialize: failed to load memory.memohood config; using defaults", exc_info=True)
            self._cfg = dict(memohood_config.DEFAULTS)

        try:
            conn = db.get_connection(hermes_home=self._hermes_home)
        except db.DbError:
            logger.error("memohood.initialize: failed to open memory.db", exc_info=True)
            self._ready = False
            return

        try:
            if not self._is_child:
                try:
                    stats = db.catch_up_from_state(conn, self._hermes_home)
                    if stats.get("indexed"):
                        logger.info(
                            "memohood: catch_up_from_state indexed %d message(s) in %d batch(es)",
                            stats["indexed"], stats["batches"],
                        )
                except Exception:  # noqa: BLE001 - a backfill hiccup must not fail agent startup
                    logger.warning("memohood.initialize: catch_up_from_state failed", exc_info=True)
        finally:
            conn.close()
        self._ready = True

    def _connect(self) -> Any:
        """Short-lived connection for the CURRENT thread (schema already ensured in initialize)."""
        return db.get_connection(hermes_home=self._hermes_home, ensure_schema=False)

    def _spawn(self, target: Any, *, name: str) -> None:
        if _spawn_context_thread is not None:
            t = _spawn_context_thread(target, name=name, daemon=True)
        else:
            t = threading.Thread(target=target, daemon=True, name=name)
        t.start()
        self._track_thread(t)

    def system_prompt_block(self) -> str:
        if self._is_child or not self._ready:
            return ""
        return _SYSTEM_PROMPT_BLOCK

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # hermes builds the tool routing table from this list in add_provider(),
        # BEFORE initialize() runs -- so it must not depend on init state, or the
        # model sees the tools but every call fails with "Unknown tool".
        if self._is_child:
            return []
        from . import tools as memohood_tools

        return memohood_tools.ALL_TOOL_SCHEMAS

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        if not self._ready:
            return "memohood: memory.db не инициализирована."
        from . import tools as memohood_tools

        session_id = kwargs.get("session_id") or self._session_id
        try:
            conn = self._connect()
        except db.DbError:
            logger.error("memohood.handle_tool_call: failed to open memory.db", exc_info=True)
            return "memohood: не удалось открыть memory.db."
        try:
            return memohood_tools.dispatch(tool_name, args, conn=conn, cfg=self._cfg, session_id=session_id)
        finally:
            conn.close()

    # -- recall -----------------------------------------------------------------

    @staticmethod
    def _normalize_query(query: str) -> str:
        terms = query_norm.meaningful_terms(query)
        return " ".join(terms) if terms else query

    def _reinforce(self, conn: Any, captures: List[Dict[str, Any]]) -> None:
        ids = [c["capture_id"] for c in captures if c.get("capture_id")]
        if not ids:
            return
        try:
            now = db.now()
            placeholders = ",".join("?" for _ in ids)
            with conn:
                conn.execute(f"UPDATE captures SET last_seen_at = ? WHERE id IN ({placeholders})", [now, *ids])
        except Exception:  # noqa: BLE001 - reinforcement is best-effort, never worth failing prefetch over
            logger.debug("memohood: failed to reinforce last_seen_at", exc_info=True)

    def _compute_prefetch_text(self, conn: Any, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return ""
        deadline = time.monotonic() + _PREFETCH_DEADLINE_S

        recall_ok, gate_score, gate_reason = gate_mod.should_recall(query, cfg=self._cfg)
        if not recall_ok:
            logger.debug("memohood.prefetch: gate skipped recall (score=%.3f, %s)", gate_score, gate_reason)
            return ""

        normalized = self._normalize_query(query)
        recall_cfg = self._cfg.get("recall") or {}
        k = int(recall_cfg.get("k", 4))
        messages_k = int(recall_cfg.get("messages_k", 2))

        try:
            captures = retrieve_mod.hybrid_search(
                conn, normalized, k, self._cfg, deadline=deadline, net_timeout=_PREFETCH_NET_TIMEOUT_S,
            )
        except Exception:  # noqa: BLE001
            logger.warning("memohood.prefetch: hybrid_search failed", exc_info=True)
            captures = []

        # v1.1: session_links BOOST + 1-hop EXPANSION. graph_rerank() already
        # degrades internally and never raises; this try/except is defense in
        # depth only, matching every other call site in this method.
        try:
            captures = graph_rerank_mod.graph_rerank(captures, db=conn, cfg=self._cfg)
        except Exception:  # noqa: BLE001
            logger.warning("memohood.prefetch: graph_rerank failed", exc_info=True)

        # v1.1: MMR + near-duplicate collapse diversity pass. Both functions
        # are pure/never-raise by contract -- no try/except needed here (see
        # post_recall.py's own module docstring "WIRING" note).
        captures = post_recall.attach_vectors(conn, captures)
        captures = post_recall.diversify(captures, cfg=self._cfg, query=normalized)

        try:
            messages = retrieve_mod.fts_search_messages(conn, normalized, messages_k)
        except Exception:  # noqa: BLE001
            logger.warning("memohood.prefetch: fts_search_messages failed", exc_info=True)
            messages = []

        if not captures and not messages:
            return ""

        self._reinforce(conn, captures)
        self._last_recall_count = len(captures) + len(messages)

        lines: List[str] = []
        if captures:
            lines.append("Воспоминания (факты/решения из прошлых разговоров):")
            for c in captures:
                tag = "[закреплено] " if c.get("pinned") else ""
                lines.append(f"- {tag}[{c.get('kind')}] {c.get('text')}")
        if messages:
            if lines:
                lines.append("")
            lines.append("Похожие места в истории диалога:")
            for m in messages:
                role = m.get("role") or "?"
                content = (m.get("content") or "")[:300]
                lines.append(f"- ({role}) {content}")
        return "\n".join(lines)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self._last_recall_count = 0
        if self._is_child or not self._ready:
            return ""
        key = session_id or self._session_id or "-"
        query = (query or "").strip()
        with self._prefetch_lock:
            cached = self._prefetch_cache.get(key)
        if cached is not None and cached[0] == query:
            self._last_recall_count = cached[2]
            return cached[1]
        try:
            conn = self._connect()
        except db.DbError:
            logger.warning("memohood.prefetch: failed to open memory.db", exc_info=True)
            return ""
        try:
            text = self._compute_prefetch_text(conn, query)
        except Exception:  # noqa: BLE001 - prefetch must never fail a turn
            logger.warning("memohood.prefetch: unexpected error", exc_info=True)
            return ""
        finally:
            conn.close()
        if not text:
            self._last_recall_count = 0
        with self._prefetch_lock:
            self._prefetch_cache[key] = (query, text, self._last_recall_count)
        return text

    def recall_status(self) -> Optional[Any]:
        if _RecallStatus is None or self._last_recall_count <= 0:
            return None
        return _RecallStatus("MemoHood", self._last_recall_count)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        # Intentionally a no-op. hermes calls this after turn N with turn N's text;
        # a result precomputed for that text was then served on turn N+1 -- recall
        # for the PREVIOUS message. memohood's recall is local and fast enough to
        # run inline in prefetch() for the actual current query.
        return

    # -- capture ------------------------------------------------------------------

    def _track_thread(self, t: threading.Thread) -> None:
        with self._bg_lock:
            self._bg_threads = [x for x in self._bg_threads if x.is_alive()]
            self._bg_threads.append(t)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if self._is_child or self._hermes_home is None:
            return
        if not (user_content or "").strip() and not (assistant_content or "").strip():
            return
        if not self._cfg.get("auto_capture", True):
            return

        hermes_home = self._hermes_home
        cfg = self._cfg
        sid = session_id or self._session_id

        def _run() -> None:
            try:
                conn = db.get_connection(hermes_home=hermes_home)
            except db.DbError:
                logger.warning("memohood.sync_turn: failed to open memory.db", exc_info=True)
                return
            try:
                capture_mod.process_turn(conn, user_content, assistant_content, session_id=sid, cfg=cfg)
            except Exception:  # noqa: BLE001 - a capture failure must never surface to the conversation loop
                logger.error("memohood.sync_turn: capture.process_turn failed", exc_info=True)
            finally:
                conn.close()

        if _HOST_RUNS_SYNC_IN_BACKGROUND:
            _run()  # already on hermes' serialized background worker
        else:
            self._spawn(_run, name="memohood-sync-turn")

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if self._is_child or not self._ready or not messages:
            return ""
        rescued: List[str] = []
        try:
            conn = self._connect()
        except db.DbError:
            logger.warning("memohood.on_pre_compress: failed to open memory.db", exc_info=True)
            return ""
        try:
            threshold = float(self._cfg.get("capture_threshold", 4.0))
            # Only DEFINITE-KEEP messages (free signal score >= threshold) -- no
            # extraction LLM call. Storing still embeds each one (and may run the
            # near-duplicate judge), so the newest few are rescued, not all.
            # Only real dialogue turns: tool/system output scored as "user" text
            # could otherwise become a pinned persona fact.
            capture_assistant = self._cfg.get("capture_assistant", True)
            candidates = []
            for msg in messages:
                role = msg.get("role")
                if role not in ("user", "assistant") or (role == "assistant" and not capture_assistant):
                    continue
                content = msg.get("content")
                if not isinstance(content, str) or not content.strip():
                    continue
                side = role
                sig = capture_mod.compute_signals(content, side=side)
                if sig["score"] >= threshold:
                    candidates.append((content, side))
            for content, side in candidates[-_MAX_PRECOMPRESS_RESCUES:]:
                result = capture_mod.extract_and_store(
                    conn, content, side=side, session_id=self._session_id, cfg=self._cfg,
                )
                if result and result.get("capture_id"):
                    rescued.append(content[:200])
        except Exception:  # noqa: BLE001
            logger.warning("memohood.on_pre_compress: rescue pass failed", exc_info=True)
        finally:
            conn.close()

        if not rescued:
            return ""
        bullets = "\n".join(f"- {r}" for r in rescued)
        return f"memohood rescued {len(rescued)} durable fact(s) into memory before this context was compressed:\n{bullets}"

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        self._join_background(timeout=5.0)

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> None:
        self._session_id = new_session_id
        if reset:
            with self._prefetch_lock:
                self._prefetch_cache.clear()

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._is_child or self._hermes_home is None or action != "add" or not content:
            return
        hermes_home = self._hermes_home
        cfg = self._cfg
        sid = self._session_id
        kind = "persona" if target == "user" else "instruction"

        def _run() -> None:
            try:
                conn = db.get_connection(hermes_home=hermes_home)
            except db.DbError:
                return
            try:
                capture_mod.manual_capture(
                    conn, content, kind=kind, notability="medium", pinned=False, session_id=sid, cfg=cfg,
                )
            except Exception:  # noqa: BLE001
                logger.debug("memohood.on_memory_write: mirror failed", exc_info=True)
            finally:
                conn.close()

        self._spawn(_run, name="memohood-memwrite")

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs: Any) -> None:
        if self._is_child or self._hermes_home is None or not task:
            return
        hermes_home = self._hermes_home
        cfg = self._cfg
        sid = self._session_id
        content = f"Делегирование ({child_session_id or '?'}): {task[:300]} -> {(result or '')[:300]}"

        def _run() -> None:
            try:
                conn = db.get_connection(hermes_home=hermes_home)
            except db.DbError:
                return
            try:
                capture_mod.manual_capture(
                    conn, content, kind="event", notability="low", pinned=False, session_id=sid, cfg=cfg,
                )
            except Exception:  # noqa: BLE001
                logger.debug("memohood.on_delegation: observation capture failed", exc_info=True)
            finally:
                conn.close()

        self._spawn(_run, name="memohood-delegation")

    # -- config / setup ---------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "gate.backend", "description": "Бэкенд гейта recall (pass = всегда вспоминать, model2vec = локальный классификатор)", "default": "pass", "choices": ["pass", "model2vec"]},
            {"key": "gate.threshold", "description": "gate: мин. похожесть на негативные примеры, чтобы реально пропустить recall", "default": 0.5},
            {"key": "gate.margin", "description": "gate: допустимое отставание позитивной похожести от негативной", "default": 0.05},
            {"key": "gate.model2vec_model", "description": "gate: модель model2vec (лениво устанавливается)", "default": "minishlab/potion-base-8M"},
            {"key": "gate.meaningful_terms_floor", "description": "gate: мин. число значимых слов в запросе, чтобы не гейтить вовсе", "default": 3},
            {"key": "post_recall.mmr.enabled", "description": "Включить MMR-разнообразие после ретрива", "default": "true", "choices": ["true", "false"]},
            {"key": "post_recall.mmr.lambda", "description": "post_recall: баланс релевантность/разнообразие (1.0 = только релевантность)", "default": 0.7},
            {"key": "post_recall.cluster.enabled", "description": "post_recall: схлопывать почти-дубликаты перед MMR", "default": "true", "choices": ["true", "false"]},
            {"key": "post_recall.cluster.threshold", "description": "post_recall: порог косинусной похожести для схлопывания дубликатов", "default": 0.93},
            {"key": "graph_rerank.enabled", "description": "Буст+добор воспоминаний по графу связанных сессий (session_links)", "default": "true", "choices": ["true", "false"]},
            {"key": "graph_rerank.max_neighbors", "description": "graph_rerank: макс. число новых воспоминаний из связанных сессий за раз", "default": 3},
            {"key": "model.provider", "description": "Провайдер LLM для извлечения/консолидации", "default": "gemini"},
            {"key": "model.model", "description": "Модель для извлечения/консолидации", "default": "gemini-3.1-flash-lite"},
            {"key": "embedder.provider", "description": "Провайдер эмбеддингов для captures", "default": "cloudflare"},
            {"key": "embedder.model", "description": "Модель эмбеддингов", "default": "@cf/baai/bge-m3"},
            {"key": "embedder.dims", "description": "Размерность эмбеддинга", "default": 1024},
            {"key": "rerank.enabled", "description": "Включить реранк Cohere", "default": "true", "choices": ["true", "false"]},
            {"key": "auto_capture", "description": "Авто-извлечение фактов из каждого хода", "default": "true", "choices": ["true", "false"]},
            {"key": "capture_assistant", "description": "Извлекать факты и из ответов агента (False = только из сообщений пользователя)", "default": "true", "choices": ["true", "false"]},
            {"key": "capture_threshold", "description": "Порог бесплатного сигнального скоринга для гарантированного захвата", "default": 4.0},
            {
                "key": "GEMINI_API_KEY", "description": "Ключ Gemini (извлечение/консолидация)", "secret": True,
                "env_var": "GEMINI_API_KEY", "url": "https://aistudio.google.com/apikey",
            },
            {"key": "CLOUDFLARE_ACCOUNT_ID", "description": "Cloudflare account id (эмбеддинги)", "secret": True, "env_var": "CLOUDFLARE_ACCOUNT_ID"},
            {"key": "CLOUDFLARE_API_TOKEN", "description": "Cloudflare Workers AI токен (эмбеддинги)", "secret": True, "env_var": "CLOUDFLARE_API_TOKEN"},
            {
                "key": "COHERE_API_KEY", "description": "Ключ Cohere (реранк)", "secret": True,
                "env_var": "COHERE_API_KEY", "url": "https://dashboard.cohere.com/api-keys",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        try:
            memohood_config.save_memohood_config_at(values, hermes_home)
        except Exception:  # noqa: BLE001 - setup-wizard writer must degrade, not crash `hermes memory setup`
            logger.error("memohood.save_config: failed to persist config", exc_info=True)

    def backup_paths(self) -> List[str]:
        # memory.db lives INSIDE HERMES_HOME (beside state.db) -- `hermes
        # backup` already walks HERMES_HOME on its own, so there is nothing
        # external to declare (see MemoryProvider.backup_paths' own
        # docstring: paths this provider stores OUTSIDE HERMES_HOME).
        return []

    def shutdown(self) -> None:
        self._join_background(timeout=3.0)
        self._ready = False

    def _join_background(self, *, timeout: float) -> None:
        with self._bg_lock:
            threads = list(self._bg_threads)
        for t in threads:
            if t.is_alive():
                t.join(timeout=timeout)
