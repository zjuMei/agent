import asyncio
import json
import unittest

import httpx
from openai import OpenAI

from ..examples.offline_demo import ScriptedLLM, calculate_call
from ..mini_agents import (
    AgentError, AsyncToolExecutor, ChainStep, Config, LLM, Message, ModelError, OutputError,
    PlanAndSolveAgent, ReActAgent, ReflectionAgent, SimpleAgent, StepLimitError,
    ToolCall, ToolChain, ToolRegistry, create_registry,
)


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.registry = create_registry()

    def test_calculator_and_rejected_expressions(self):
        self.assertEqual(self.registry.execute("calculator", {"expression": "sqrt(16) + 2 * 3"}),
                         {"ok": True, "result": 10})
        for expression in ["__import__('os').getcwd()", "True + 1", "2 ** 101", "1 / 0", "(-1)**0.5"]:
            with self.subTest(expression=expression):
                self.assertFalse(self.registry.execute("calculator", {"expression": expression})["ok"])

    def test_schema_and_validation_before_execution(self):
        calls = []
        @self.registry.tool
        def add(a: int, b: int = 1) -> int:
            """相加。"""
            calls.append(a)
            return a + b
        schema = self.registry.schemas()[-1]["function"]["parameters"]
        self.assertEqual(schema["required"], ["a"])
        self.assertFalse(schema["additionalProperties"])
        for arguments in ['{"a":"3"}', '{"a":true}', '{"a":1,"typo":2}', '{}', '[]', '{bad']:
            self.assertFalse(self.registry.execute("add", arguments)["ok"])
        self.assertEqual(calls, [])
        self.assertEqual(self.registry.execute("add", {"a": 3})["result"], 4)

    def test_unknown_duplicate_and_runtime_failure(self):
        self.assertFalse(self.registry.execute("missing", {})["ok"])
        with self.assertRaises(ValueError):
            self.registry.register(self.registry._tools["calculator"])
        @self.registry.tool
        def broken() -> int:
            raise RuntimeError("测试异常")
        self.assertIn("测试异常", self.registry.execute("broken", {})["error"])

    def test_chain_passes_result_and_stops_on_error(self):
        chain = ToolChain(self.registry, [
            ChainStep("calculator", {"expression": "{input} * 2"}, "first"),
            ChainStep("calculator", {"expression": "{first} + 1"}, "answer"),
        ])
        self.assertEqual(chain.run(input=3)["answer"], 7)
        with self.assertRaises(AgentError):
            ToolChain(self.registry, [ChainStep("calculator", {"expression": "1 / 0"}, "bad")]).run()

    def test_async_execution_order_and_failure_isolation(self):
        result = asyncio.run(AsyncToolExecutor(self.registry).run([
            ("calculator", {"expression": "1 + 2"}), ("missing", {}),
            ("calculator", {"expression": "2 * 3"}),
        ]))
        self.assertEqual([r.get("result") for r in result], [3, None, 6])
        self.assertFalse(result[1]["ok"])


class AgentTests(unittest.TestCase):
    def test_simple_history_clear_and_defensive_copy(self):
        llm = ScriptedLLM(["第一答", "第二答", "第三答"])
        agent = SimpleAgent(llm, max_turns=1)
        agent.run("第一问")
        history = agent.get_history()
        history[0].content = "被修改"
        agent.run("第二问")
        self.assertEqual(llm.requests[1][0][1]["content"], "第一问")
        self.assertEqual([m.content for m in agent.get_history()], ["第二问", "第二答"])
        agent.clear_history()
        agent.run("第三问")
        self.assertEqual(len(llm.requests[2][0]), 2)

    def test_react_tool_protocol_and_complete_turn_trimming(self):
        llm = ScriptedLLM([calculate_call(), "14", "后续回答"])
        agent = ReActAgent(llm, tools=create_registry(), max_turns=1)
        self.assertEqual(agent.run("算一下"), "14")
        messages, schemas = llm.requests[1]
        self.assertTrue(schemas)
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertEqual(messages[-1]["tool_call_id"], messages[-2]["tool_calls"][0]["id"])
        self.assertEqual(json.loads(messages[-1]["content"])["result"], 14)
        agent.run("继续")
        self.assertEqual([m.role for m in agent.get_history()], ["user", "assistant"])

    def test_tool_error_is_visible_to_model(self):
        llm = ScriptedLLM([calculate_call("1 / 0"), "无法除以零。"])
        agent = ReActAgent(llm, tools=create_registry())
        agent.run("除以零")
        self.assertFalse(json.loads(llm.requests[1][0][-1]["content"])["ok"])

    def test_unknown_tool_does_not_execute_arbitrary_function(self):
        call = Message("assistant", tool_calls=[ToolCall("x", "unknown", "{}")])
        llm = ScriptedLLM([call, "工具不可用"])
        ReActAgent(llm, tools=create_registry()).run("任务")
        self.assertIn("未注册", llm.requests[1][0][-1]["content"])

    def test_limit_preserves_previous_history(self):
        llm = ScriptedLLM(["第一轮成功", calculate_call()])
        agent = ReActAgent(llm, tools=create_registry(), max_steps=1)
        agent.run("第一问")
        with self.assertRaises(StepLimitError):
            agent.run("第二问")
        self.assertEqual(len(agent.get_history()), 2)
        self.assertEqual(agent.trace[-1]["result"], 14)

    def test_invalid_multiple_calls_are_rejected_before_execution(self):
        first = calculate_call().tool_calls[0]
        for calls in [[first, first], [first, ToolCall("other", "calculator", first.arguments)]]:
            agent = ReActAgent(ScriptedLLM([Message("assistant", tool_calls=calls)]), tools=create_registry())
            with self.assertRaises(OutputError):
                agent.run("计算")
            self.assertFalse(any(event["stage"] == "工具" for event in agent.trace))
            self.assertEqual(agent.get_history(), [])

    def test_simple_cannot_execute_tool_calls(self):
        with self.assertRaises(OutputError):
            SimpleAgent(ScriptedLLM([calculate_call()])).run("计算")

    def test_empty_response_or_input_not_saved(self):
        agent = SimpleAgent(ScriptedLLM([" "]))
        for query in ["", "问题"]:
            with self.assertRaises((ValueError, OutputError)):
                agent.run(query)
        self.assertEqual(agent.get_history(), [])

    def test_reflection_revision_and_early_approval(self):
        llm = ScriptedLLM(["草稿", '{"approved":false,"feedback":"补充例子"}', "修订稿",
                           '{"approved":true,"feedback":""}'])
        agent = ReflectionAgent(llm)
        self.assertEqual(agent.run("写说明"), "修订稿")
        self.assertEqual([e["stage"] for e in agent.trace], ["草稿", "评审", "修订", "评审"])
        self.assertEqual(len(agent.get_history()), 2)
        self.assertTrue(all(tools is None for _, tools in llm.requests))

    def test_reflection_requires_boolean_and_does_not_match_substrings(self):
        for review in ['{"approved":"false","feedback":""}', '不是 JSON',
                       '{"feedback":"无需改进并不代表通过"}']:
            with self.subTest(review=review), self.assertRaises(OutputError):
                ReflectionAgent(ScriptedLLM(["草稿", review])).run("问题")

    def test_plan_execution_tools_and_summary(self):
        llm = ScriptedLLM(['```json\n["计算"]\n```', calculate_call(), "14", "最终答案 14"])
        agent = PlanAndSolveAgent(llm, tools=create_registry())
        self.assertEqual(agent.run("计算表达式"), "最终答案 14")
        self.assertIsNone(llm.requests[0][1])
        self.assertTrue(llm.requests[1][1])
        self.assertIsNone(llm.requests[-1][1])
        self.assertEqual(len(agent.get_history()), 2)

    def test_invalid_or_overlong_plan_never_executes_steps(self):
        for plan in ['[]', '["a","b"]', '[1]', '{"step":"a"}', '[" "]', 'print(1)']:
            llm = ScriptedLLM([plan])
            with self.subTest(plan=plan), self.assertRaises(OutputError):
                PlanAndSolveAgent(llm, max_plan_steps=1).run("问题")
            self.assertEqual(len(llm.requests), 1)


class AdapterTests(unittest.TestCase):
    def adapter(self, payload, status=200):
        llm = LLM(Config(model="test", api_key="test-key"))
        llm.client.close()
        self.requests = []
        def respond(request):
            self.requests.append(json.loads(request.content))
            return httpx.Response(status, json=payload)
        llm.client = OpenAI(api_key="test-key", max_retries=0,
                            http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        self.addCleanup(llm.close)
        return llm

    def response(self, message, reason="stop"):
        return {"id": "test", "object": "chat.completion", "created": 0, "model": "test",
                "choices": [{"index": 0, "message": message, "finish_reason": reason}]}

    def test_real_sdk_serialization_and_tool_call_parsing(self):
        llm = self.adapter(self.response({"role": "assistant", "content": None,
            "tool_calls": [calculate_call().tool_calls[0].to_dict()]}, "tool_calls"))
        reply = llm.chat([Message("user", "算一下")], tools=create_registry().schemas())
        self.assertEqual(reply.tool_calls[0].name, "calculator")
        self.assertFalse(self.requests[0]["parallel_tool_calls"])
        self.assertNotIn("enable_thinking", self.requests[0])
        self.assertEqual(llm.calls, 1)

    def test_incomplete_reply_never_reaches_execution(self):
        llm = self.adapter(self.response({"role": "assistant", "content": "半句话"}, "length"))
        with self.assertRaises(OutputError):
            SimpleAgent(llm).run("问题")

    def test_api_error_is_sanitized(self):
        llm = self.adapter({"error": {"message": "sensitive-value", "type": "invalid_request_error"}}, 401)
        with self.assertRaises(ModelError) as result:
            llm.chat([Message("user", "问题")])
        self.assertNotIn("sensitive-value", str(result.exception))


if __name__ == "__main__":
    unittest.main()
