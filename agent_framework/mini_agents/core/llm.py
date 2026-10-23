"""唯一接触 OpenAI SDK 的地方；Agent 只依赖 chat 接口。"""

from time import perf_counter
from typing import Callable, Protocol

from httpx import HTTPError
from openai import OpenAI, OpenAIError

from .config import Config
from .exceptions import ModelError, OutputError
from .message import Message, ToolCall


class ChatModel(Protocol):
    def chat(self, messages: list[Message], *, tools: list | None = None,
             stream: bool = True, on_token: Callable[[str], None] | None = None) -> Message: ...


class LLM:
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url,
                             timeout=config.timeout, max_retries=1)
        self.calls = 0
        self.seconds = 0.0

    def chat(self, messages: list[Message], *, tools: list | None = None,
             stream: bool = True, on_token: Callable[[str], None] | None = None) -> Message:
        """默认使用流式请求；实时通知文本片段，最终仍返回完整 Message。"""
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
                max_completion_tokens=self.config.max_tokens, stream=stream, **options,
            )
            if stream:
                message, reason = self._read_stream(response, on_token)
            else:
                if not response.choices:
                    raise OutputError("模型没有返回候选结果")
                choice = response.choices[0]
                if choice.message.refusal:
                    raise OutputError("模型拒绝处理本次请求")
                message, reason = self._message(choice.message), choice.finish_reason
            if reason not in ("stop", "tool_calls"):
                raise OutputError(f"模型未正常完成：{reason}")
            if bool(message.tool_calls) != (reason == "tool_calls"):
                raise OutputError("模型结束标记与工具调用不一致，未执行工具")
            return message
        except (OpenAIError, HTTPError) as error:
            # 避免把服务商响应中可能包含的凭证或内部地址输出给终端。
            raise ModelError(f"模型调用失败：{type(error).__name__}；请检查服务、网络和配置。") from None
        finally:
            self.seconds += perf_counter() - start

    @staticmethod
    def _message(raw) -> Message:
        calls = []
        for call in raw.tool_calls or []:
            if call.type != "function":
                raise OutputError("只支持 function 类型工具调用")
            calls.append(ToolCall(call.id, call.function.name, call.function.arguments))
        return Message("assistant", raw.content or "", calls)

    @staticmethod
    def _read_stream(response, on_token) -> tuple[Message, str | None]:
        content, calls, reason = [], {}, None
        # 正常结束、解析失败、回调异常或用户中断时，都关闭 HTTP 流。
        with response:
            for chunk in response:
                if not chunk.choices:  # 部分服务会额外返回仅含 usage 的片段。
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason is not None:
                    reason = choice.finish_reason
                delta = choice.delta
                if delta.refusal:
                    raise OutputError("模型拒绝处理本次请求")
                if delta.content:
                    content.append(delta.content)
                    if on_token is not None:
                        on_token(delta.content)
                for part in delta.tool_calls or []:
                    if type(part.index) is not int or part.index < 0 or part.type not in (None, "function"):
                        raise OutputError("工具调用片段的序号或类型无效")
                    call = calls.setdefault(part.index, {"id": "", "name": "", "arguments": ""})
                    call["id"] += part.id or ""
                    if part.function is not None:
                        call["name"] += part.function.name or ""
                        call["arguments"] += part.function.arguments or ""
        if sorted(calls) != list(range(len(calls))):
            raise OutputError("工具调用片段缺少连续序号")
        complete = [ToolCall(**calls[index]) for index in sorted(calls)]
        if any(not call.id or not call.name for call in complete):
            raise OutputError("工具调用未完整接收，缺少 ID 或名称")
        return Message("assistant", "".join(content), complete), reason

    def close(self):
        self.client.close()
