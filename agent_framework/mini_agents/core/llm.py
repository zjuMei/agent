"""唯一接触 OpenAI SDK 的地方；Agent 只依赖 chat 接口。"""

from time import perf_counter
from typing import Protocol

from openai import OpenAI, OpenAIError

from .config import Config
from .exceptions import ModelError, OutputError
from .message import Message, ToolCall


class ChatModel(Protocol):
    def chat(self, messages: list[Message], *, tools: list | None = None) -> Message: ...


class LLM:
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url,
                             timeout=config.timeout, max_retries=1)
        self.calls = 0
        self.seconds = 0.0

    def chat(self, messages: list[Message], *, tools: list | None = None) -> Message:
        options = {}
        if tools:
            options.update(tools=tools, tool_choice="auto", parallel_tool_calls=False)
        if self.config.enable_thinking is not None:
            options["extra_body"] = {"enable_thinking": self.config.enable_thinking}
        start = perf_counter()
        self.calls += 1
        try:
            response = self.client.chat.completions.create(
                model=self.config.model, messages=[m.to_dict() for m in messages],
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_tokens, **options,
            )
        except OpenAIError as error:
            # 避免把服务商响应中可能包含的凭证或内部地址输出给终端。
            raise ModelError(f"模型调用失败：{type(error).__name__}；请检查服务、网络和配置。") from None
        finally:
            self.seconds += perf_counter() - start
        if not response.choices:
            raise OutputError("模型没有返回候选结果")
        choice = response.choices[0]
        if choice.finish_reason not in ("stop", "tool_calls") or choice.message.refusal:
            raise OutputError(f"模型未正常完成：{choice.finish_reason}")
        raw = choice.message
        calls = []
        for call in raw.tool_calls or []:
            if call.type != "function":
                raise OutputError("只支持 function 类型工具调用")
            calls.append(ToolCall(call.id, call.function.name, call.function.arguments))
        return Message("assistant", raw.content or "", calls)

    def close(self):
        self.client.close()
