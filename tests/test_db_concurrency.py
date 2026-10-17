"""db.py robustness: concurrent catch-up, idempotent meta defaults, conn cleanup."""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest


def _make_state_db(path, n):
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, "
        "content TEXT, timestamp REAL, active INTEGER)"
    )
    now = time.time()
    conn.executemany(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?, 1)",
        [(i, "s1", "user" if i % 2 else "assistant", f"сообщение номер {i}", now) for i in range(1, n + 1)],
    )
    conn.commit()
    conn.close()


def test_concurrent_catch_up_indexes_each_message_once(memohood):
    home = memohood._hermes_home_for_test
    _make_state_db(home / "state.db", 1200)
    memohood.db.get_connection(hermes_home=str(home)).close()  # schema first, like initialize()

    errors = []
    barrier = threading.Barrier(3)

    def worker():
        conn = memohood.db.get_connection(hermes_home=str(home), ensure_schema=False)
        try:
            barrier.wait()
            memohood.db.catch_up_from_state(conn, str(home), batch_size=50)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            conn.close()

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors
    conn = memohood.db.get_connection(hermes_home=str(home), ensure_schema=False)
    rows = conn.execute("SELECT message_id, COUNT(*) AS n FROM messages_fts GROUP BY message_id").fetchall()
    assert len(rows) == 1200
    assert all(r["n"] == 1 for r in rows)
    conn.close()


def test_meta_defaults_idempotent(memohood):
    conn = memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
    memohood.db._ensure_meta_defaults(conn)
    memohood.db._ensure_meta_defaults(conn)
    assert conn.execute("SELECT COUNT(*) FROM _meta WHERE key='last_indexed_message_id'").fetchone()[0] == 1
    conn.close()


def test_get_connection_closes_on_schema_failure(memohood, monkeypatch):
    opened = []
    real_connect = memohood.db.connect

    def spy_connect(*a, **kw):
        c = real_connect(*a, **kw)
        opened.append(c)
        return c

    def boom(conn):
        raise memohood.db.DbError("schema broken")

    monkeypatch.setattr(memohood.db, "connect", spy_connect)
    monkeypatch.setattr(memohood.db, "init_schema", boom)
    with pytest.raises(memohood.db.DbError):
        memohood.db.get_connection(hermes_home=str(memohood._hermes_home_for_test))
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute("SELECT 1")  # closed
