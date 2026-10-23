"""按用户、会话和 Agent 类型保存完整成功轮次，读取时才限制上下文。"""

import json
from dataclasses import asdict

from ..core.exceptions import AgentError
from ..core.message import Message, ToolCall


class SessionStore:
    def __init__(self, db, *, user_id="local", session_id="default"):
        if not user_id.strip() or not session_id.strip():
            raise ValueError("user_id 和 session_id 不能为空")
        self.db = db
        self.user_id, self.session_id = user_id, session_id

    def load(self, agent_type, *, limit=5):
        if type(limit) is not int or limit < 1:
            raise ValueError("limit 必须是正整数")
        with self.db.transaction() as connection:
            rows = connection.execute(
                "SELECT messages FROM turns WHERE user_id=? AND session_id=? AND agent_type=? "
                "ORDER BY id DESC LIMIT ?",
                (self.user_id, self.session_id, agent_type, limit),
            ).fetchall()
        try:
            turns = []
            for row in reversed(rows):
                turn = []
                data = json.loads(row["messages"])
                if not isinstance(data, list) or not data:
                    raise ValueError("轮次必须是非空消息数组")
                for item in data:
                    if not isinstance(item, dict):
                        raise ValueError("消息必须是对象")
                    calls = [ToolCall(**call) for call in item.pop("tool_calls", [])]
                    turn.append(Message(**item, tool_calls=calls))
                turns.append(turn)
            return turns
        except (ValueError, TypeError, KeyError) as error:
            raise AgentError("会话数据损坏，无法恢复") from error

    def append(self, agent_type, messages):
        payload = json.dumps([asdict(message) for message in messages], ensure_ascii=False)
        with self.db.transaction() as connection:
            connection.execute(
                "INSERT INTO turns(user_id, session_id, agent_type, messages) VALUES (?, ?, ?, ?)",
                (self.user_id, self.session_id, agent_type, payload),
            )

    def clear(self, agent_type):
        with self.db.transaction() as connection:
            connection.execute(
                "DELETE FROM turns WHERE user_id=? AND session_id=? AND agent_type=?",
                (self.user_id, self.session_id, agent_type),
            )
