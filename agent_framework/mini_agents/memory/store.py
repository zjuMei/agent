"""显式维护的长期记忆；不会把聊天或模型猜测自动写成事实。"""

from datetime import datetime, timezone
from uuid import uuid4

from ..core.retrieval import bounded, rank


def memory_content(content):
    if not isinstance(content, str) or not content.strip() or len(content) > 2000:
        raise ValueError("记忆内容必须是 1 至 2000 字符的非空文本")
    return content.strip()


class MemoryStore:
    def __init__(self, db, *, user_id="local"):
        if not user_id.strip():
            raise ValueError("user_id 不能为空")
        self.db, self.user_id = db, user_id

    def add(self, content):
        content = memory_content(content)
        now = datetime.now(timezone.utc).isoformat()
        with self.db.transaction() as connection:
            connection.execute(
                "INSERT INTO memories VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id, content) DO NOTHING",
                (uuid4().hex, self.user_id, content, now, now),
            )
            row = connection.execute(
                "SELECT * FROM memories WHERE user_id=? AND content=?", (self.user_id, content),
            ).fetchone()
        return dict(row)

    def list(self, *, limit=10):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit 必须是 1 至 100 的整数")
        with self.db.transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM memories WHERE user_id=? ORDER BY updated_at DESC, id LIMIT ?",
                (self.user_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def search(self, query, *, limit=3, max_chars=2000):
        with self.db.transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM memories WHERE user_id=? ORDER BY updated_at DESC, id", (self.user_id,),
            ).fetchall()
        return bounded(rank(query, [dict(row) for row in rows]), limit=limit, max_chars=max_chars)

    def context(self, query, *, max_chars=2000):
        # 相关记忆优先，再补最近记忆，让不含相同关键词的偏好也有机会进入上下文。
        items = {item["id"]: item for item in self.search(query)}
        for item in self.list(limit=3):
            items.setdefault(item["id"], item)
        return bounded(list(items.values()), limit=6, max_chars=max_chars)

    def update(self, memory_id, content):
        content = memory_content(content)
        with self.db.transaction() as connection:
            return connection.execute(
                "UPDATE memories SET content=?, updated_at=? WHERE id=? AND user_id=?",
                (content, datetime.now(timezone.utc).isoformat(), memory_id, self.user_id),
            ).rowcount == 1

    def delete(self, memory_id):
        with self.db.transaction() as connection:
            return connection.execute(
                "DELETE FROM memories WHERE id=? AND user_id=?", (memory_id, self.user_id),
            ).rowcount == 1
