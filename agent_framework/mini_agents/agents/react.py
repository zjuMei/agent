import json
from time import perf_counter

from ..core.agent import Agent
from ..core.exceptions import OutputError, StepLimitError
from ..core.message import Message
from ..tools.registry import ToolRegistry


class ReActAgent(Agent):
    """循环：模型选择行动 → 程序执行工具 → 模型读取结果，直到返回文本。"""

    def __init__(self, llm, *, tools=None, max_steps=6, **kwargs):
        super().__init__(llm, **kwargs)
        if max_steps < 1:
            raise ValueError("max_steps 必须大于 0")
        self.tools = tools if tools is not None else ToolRegistry()
        self.max_steps = max_steps

    def _tool_loop(self, messages, *, stage="执行"):
        schemas = self.tools.schemas()
        for _ in range(self.max_steps):
            reply = self._chat(messages, tools=schemas, stage=stage)
            # 必须在执行前检查整条回复，避免部分执行后才发现协议错误。
            if len(reply.tool_calls) > 1:
                raise OutputError("教学版每次只接受一个工具调用，本次未执行工具")
            messages.append(reply)
            if not reply.tool_calls:
                return reply.content
            call = reply.tool_calls[0]
            start = perf_counter()
            result = self.tools.execute(call.name, call.arguments)
            self.trace.append({"stage": "工具", "name": call.name, "arguments": call.arguments,
                               "seconds": round(perf_counter() - start, 3), **result})
            messages.append(Message("tool", json.dumps(result, ensure_ascii=False), tool_call_id=call.id))
        raise StepLimitError(f"达到 {self.max_steps} 次模型调用上限，尚未得到最终回答")

    def _run(self, messages):
        return self._tool_loop(messages)
