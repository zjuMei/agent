"""用真正的 SDK 读取模拟 SSE，验证片段时序、调用拼接和流中断。"""

import contextlib
import io
import json
import unittest

import httpx
from openai import OpenAI

from ..demo import StreamPrinter
from ..examples.offline_demo import ScriptedLLM, calculate_call
from ..mini_agents import (
    Config, LLM, Message, ModelError, OutputError, PlanAndSolveAgent, ReActAgent,
    ReflectionAgent, SimpleAgent, ToolRegistry,
)


def chunk(delta=None, finish=None):
    return {"id": "test", "object": "chat.completion.chunk", "created": 0, "model": "test",
            "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}


def tool_part(arguments, *, index=0, id=None, name=None):
    part = {"index": index, "function": {"arguments": arguments}}
    if id is not None:
        part.update(id=id, type="function")
    if name is not None:
        part["function"]["name"] = name
    return chunk({"tool_calls": [part]})


class SSEStream(httpx.SyncByteStream):
    def __init__(self, frames, *, disconnect=False):
        self.frames = frames
        self.disconnect = disconnect
        self.sent = 0
        self.closed = False

    def __iter__(self):
        for frame in self.frames:
            self.sent += 1
            yield ("data: " + json.dumps(frame, ensure_ascii=False) + "\n\n").encode()
        if self.disconnect:
            raise httpx.ReadError("模拟流中断")
        yield b"data: [DONE]\n\n"

    def close(self):
        self.closed = True


class StreamingTests(unittest.TestCase):
    def adapter(self, *sources):
        llm = LLM(Config(model="test", api_key="test-key"))
        llm.client.close()
        self.requests = []
        pending = iter(sources)

        def respond(request):
            self.requests.append(json.loads(request.content))
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=next(pending))

        llm.client = OpenAI(api_key="test-key", max_retries=0,
                            http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        self.addCleanup(llm.close)
        return llm

    def test_default_stream_delivers_text_before_completion_and_saves_full_history(self):
        source = SSEStream([
            chunk({"role": "assistant", "content": "你好"}), chunk({"content": "世界"}),
            chunk(finish="stop"), {"choices": [], "usage": {"total_tokens": 3}},
        ])
        llm = self.adapter(source)
        agent = SimpleAgent(llm)
        deltas = []

        def observe(event):
            if event["type"] == "text_delta":
                deltas.append((event["text"], source.sent))
                self.assertFalse(source.closed)

        self.assertEqual(agent.run("问候", on_event=observe), "你好世界")
        self.assertEqual(deltas, [("你好", 1), ("世界", 2)])
        self.assertTrue(self.requests[0]["stream"])
        self.assertTrue(source.closed)
        self.assertEqual(agent.get_history()[-1].content, "你好世界")

    def test_fragmented_tool_executes_only_after_full_stream_is_closed(self):
        first = SSEStream([
            tool_part('{"val', id="call_", name="dou"),
            tool_part('ue":', id="1", name="ble"), tool_part('3}'), chunk(finish="tool_calls"),
        ])
        second = SSEStream([chunk({"content": "结果是"}), chunk({"content": "6"}), chunk(finish="stop")])
        llm = self.adapter(first, second)
        registry = ToolRegistry()
        effects = []

        @registry.tool
        def double(value: int) -> int:
            self.assertTrue(first.closed)
            self.assertEqual(first.sent, len(first.frames))
            effects.append(value)
            return value * 2

        agent = ReActAgent(llm, tools=registry)
        events = []
        self.assertEqual(agent.run("求两倍", on_event=events.append), "结果是6")
        self.assertEqual(effects, [3])
        messages = self.requests[1]["messages"]
        call = messages[-2]["tool_calls"][0]
        self.assertEqual(call["id"], "call_1")
        self.assertEqual(call["function"], {"name": "double", "arguments": '{"value":3}'})
        self.assertEqual(messages[-1]["tool_call_id"], "call_1")
        self.assertEqual(json.loads(messages[-1]["content"]), {"ok": True, "result": 6})
        self.assertEqual([e["type"] for e in events], [
            "model_start", "model_end", "tool_result", "model_start", "text_delta", "text_delta", "model_end",
        ])
        self.assertTrue(second.closed)

    def test_missing_finish_or_truncation_never_executes_tool_or_commits_history(self):
        for reason in [None, "length", "content_filter", "stop"]:
            frames = [tool_part('{"value":3}', id="x", name="double")]
            if reason is not None:
                frames.append(chunk(finish=reason))
            source = SSEStream(frames)
            llm = self.adapter(source)
            registry, effects = ToolRegistry(), []

            @registry.tool
            def double(value: int) -> int:
                effects.append(value)
                return value * 2

            agent = ReActAgent(llm, tools=registry)
            with self.subTest(reason=reason), self.assertRaises(OutputError):
                agent.run("计算")
            self.assertEqual(effects, [])
            self.assertEqual(agent.get_history(), [])
            self.assertTrue(source.closed)

    def test_disconnect_after_partial_text_is_reported_without_replay(self):
        source = SSEStream([chunk({"content": "部分回答"})], disconnect=True)
        llm = self.adapter(source)
        agent, events = SimpleAgent(llm), []
        with self.assertRaises(ModelError):
            agent.run("问题", on_event=events.append)
        self.assertTrue(any(e["type"] == "text_delta" for e in events))
        self.assertFalse(any(e["type"] == "model_end" for e in events))
        self.assertEqual(agent.get_history(), [])
        self.assertEqual(len(self.requests), 1)
        self.assertTrue(source.closed)

    def test_refusal_closes_stream_and_does_not_commit(self):
        source = SSEStream([chunk({"refusal": "无法处理"}), chunk(finish="stop")])
        agent = SimpleAgent(self.adapter(source))
        with self.assertRaises(OutputError):
            agent.run("问题")
        self.assertTrue(source.closed)
        self.assertEqual(agent.get_history(), [])

    def test_multiple_tool_calls_rejected_before_any_side_effect(self):
        source = SSEStream([
            tool_part('{"value":1}', id="a", name="double"),
            tool_part('{"value":2}', index=1, id="b", name="double"), chunk(finish="tool_calls"),
        ])
        registry, effects = ToolRegistry(), []

        @registry.tool
        def double(value: int) -> int:
            effects.append(value)
            return value * 2

        agent = ReActAgent(self.adapter(source), tools=registry)
        with self.assertRaises(OutputError):
            agent.run("计算")
        self.assertEqual(effects, [])
        self.assertTrue(source.closed)

    def test_callback_failure_and_keyboard_interrupt_close_stream_and_reset_run_state(self):
        for exception in [ValueError, KeyboardInterrupt]:
            first = SSEStream([chunk({"content": "片段"}), chunk(finish="stop")])
            second = SSEStream([chunk({"content": "后续回答"}), chunk(finish="stop")])
            agent = SimpleAgent(self.adapter(first, second))

            def interrupt(event):
                if event["type"] == "text_delta":
                    raise exception()

            with self.subTest(exception=exception), self.assertRaises(exception):
                agent.run("第一问", on_event=interrupt)
            self.assertTrue(first.closed)
            self.assertEqual(agent.get_history(), [])
            self.assertIsNone(agent._on_event)
            self.assertEqual(agent.run("第二问"), "后续回答")

    def test_invalid_tool_metadata_rejected(self):
        for frames in [
            [tool_part("{}", name="missing_id"), chunk(finish="tool_calls")],
            [tool_part("{}", id="x"), chunk(finish="tool_calls")],
            [tool_part("{}", index=1, id="x", name="tool"), chunk(finish="tool_calls")],
        ]:
            source = SSEStream(frames)
            with self.subTest(frames=frames), self.assertRaises(OutputError):
                self.adapter(source).chat([Message("user", "问题")])
            self.assertTrue(source.closed)


class AgentEventTests(unittest.TestCase):
    def test_all_four_modes_deliver_stage_events_and_keep_return_values(self):
        from ..mini_agents import create_registry
        cases = [
            (SimpleAgent, ["你好"], {}, "你好"),
            (ReActAgent, [calculate_call(), "14"], {"tools": create_registry()}, "14"),
            (ReflectionAgent, ["草稿", '{"approved":false,"feedback":"修订"}', "修订稿",
                               '{"approved":true,"feedback":""}'], {}, "修订稿"),
            (PlanAndSolveAgent, ['["计算"]', calculate_call(), "14", "最终14"],
             {"tools": create_registry()}, "最终14"),
        ]
        for cls, replies, options, expected in cases:
            events = []
            agent = cls(ScriptedLLM(replies), **options)
            with self.subTest(mode=cls.__name__):
                self.assertEqual(agent.run("问题", on_event=events.append), expected)
                starts = [e["stage"] for e in events if e["type"] == "model_start"]
                ends = [e["stage"] for e in events if e["type"] == "model_end"]
                self.assertEqual(starts, ends)
                self.assertTrue(any(e["type"] == "text_delta" for e in events))
                self.assertEqual(agent.get_history()[-1].content, expected)

    def test_non_stream_mode_does_not_emit_text_fragments(self):
        events = []
        agent = SimpleAgent(ScriptedLLM(["完整回答"]))
        self.assertEqual(agent.run("问题", stream=False, on_event=events.append), "完整回答")
        self.assertFalse(any(e["type"] == "text_delta" for e in events))

    def test_console_does_not_duplicate_simple_answer_and_labels_reflection_final_answer(self):
        for agent, answer in [
            (SimpleAgent(ScriptedLLM(["你好世界"])), "你好世界"),
            (ReflectionAgent(ScriptedLLM(["最终稿", '{"approved":true,"feedback":""}'])), "最终稿"),
        ]:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                printer = StreamPrinter()
                self.assertEqual(agent.run("问题", on_event=printer), answer)
                printer.finish(answer)
            if isinstance(agent, SimpleAgent):
                self.assertEqual(output.getvalue().count(answer), 1)
            else:
                self.assertIn("助手：最终稿", output.getvalue())


if __name__ == "__main__":
    unittest.main()
