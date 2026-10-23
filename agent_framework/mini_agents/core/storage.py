"""一个 SQLite 文件保存会话、记忆和文档；事务也可供工具线程安全使用。"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

from .exceptions import AgentError


class SQLiteStore:
    def __init__(self, path):
        if str(path) != ":memory:":
            path = Path(path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        try:
            self._connection = sqlite3.connect(str(path), check_same_thread=False)
        except sqlite3.Error as error:
            raise AgentError(f"无法打开本地数据库：{error}") from error
        self._connection.row_factory = sqlite3.Row
        try:
            with self._connection:
                self._connection.executescript("""
                    CREATE TABLE IF NOT EXISTS turns (
                        id INTEGER PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        agent_type TEXT NOT NULL,
                        messages TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS turns_scope
                        ON turns(user_id, session_id, agent_type, id);
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(user_id, content)
                    );
                    CREATE TABLE IF NOT EXISTS chunks (
                        namespace TEXT NOT NULL,
                        source TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        start_line INTEGER NOT NULL,
                        end_line INTEGER NOT NULL,
                        PRIMARY KEY(namespace, chunk_id)
                    );
                    CREATE INDEX IF NOT EXISTS chunks_source ON chunks(namespace, source);
                """)
        except sqlite3.Error as error:
            self._connection.close()
            raise AgentError(f"无法初始化本地数据库：{error}") from error
        except BaseException:
            self._connection.close()
            raise

    @contextmanager
    def transaction(self):
        try:
            with self._lock, self._connection:
                yield self._connection
        except sqlite3.Error as error:
            raise AgentError(f"本地数据库操作失败：{error}") from error

    def close(self):
        with self._lock:
            self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
