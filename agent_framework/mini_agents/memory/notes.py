"""每个用户/会话一份当前任务笔记，明确更新整份内容，支持 Markdown 正文。"""

from datetime import datetime, timezone


class NoteStore:
    def __init__(self, db, *, user_id="local", session_id="default"):
        if not user_id.strip() or not session_id.strip():
            raise ValueError("user_id 和 session_id 不能为空")
        self.db, self.user_id, self.session_id = db, user_id, session_id

    def get(self):
        with self.db.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM task_notes WHERE user_id=? AND session_id=?",
                (self.user_id, self.session_id),
            ).fetchone()
        return dict(row) if row is not None else None

    def save(self, content):
        if not isinstance(content, str) or not content.strip() or len(content) > 2000:
            raise ValueError("任务笔记必须是 1 至 2000 字符的非空文本")
        item = {"user_id": self.user_id, "session_id": self.session_id,
                "content": content.strip(), "updated_at": datetime.now(timezone.utc).isoformat()}
        with self.db.transaction() as connection:
            connection.execute(
                "INSERT INTO task_notes VALUES (:user_id, :session_id, :content, :updated_at) "
                "ON CONFLICT(user_id, session_id) DO UPDATE SET "
                "content=excluded.content, updated_at=excluded.updated_at", item,
            )
        return item

    def clear(self):
        with self.db.transaction() as connection:
            return connection.execute(
                "DELETE FROM task_notes WHERE user_id=? AND session_id=?",
                (self.user_id, self.session_id),
            ).rowcount == 1
