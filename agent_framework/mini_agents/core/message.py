"""保留 Function Calling 的调用 ID，使工具结果与请求一一对应。"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    def to_dict(self):
        return {"id": self.id, "type": "function", "function": {
            "name": self.name, "arguments": self.arguments,
        }}


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None

    def __post_init__(self):
        if self.role not in ("system", "user", "assistant", "tool"):
            raise ValueError("未知消息角色")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("只有 assistant 消息可以发起工具调用")
        if (self.role == "tool") != bool(self.tool_call_id):
            raise ValueError("只有 tool 消息必须提供 tool_call_id")

    def to_dict(self):
        data = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        return data
