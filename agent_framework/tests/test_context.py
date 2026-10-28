"""验证实际发送内容、工具协议、预算失败和原始数据保留，不联网。"""

from copy import deepcopy
import contextlib
import io
import json
import unittest
from unittest.mock import patch

from ..demo import main
from ..examples.offline_demo import ScriptedLLM, calculate_call
from ..mini_agents import (
    ContextBuilder, ContextConfig, ContextLimitError, MemoryStore, Message, OutputError,
    PlanAndSolveAgent, ReActAgent, ReflectionAgent, SessionStore, SimpleAgent, SQLiteStore,
    ToolCall, create_registry,
)
from ..mini_agents.core.context import compact_tool_result


class BuilderTests(unittest.TestCase):
    def test_budget_counts_message_structure_tool_arguments_and_schemas(self):
        messages = [Message("system", "规则"), Message("user", "任务")]
        tools = create_registry().schemas()
        total = ContextBuilder.measure(messages, tools)
        self.assertGreater(total, sum(len(message.content) for message in messages))
        self.assertGreater(total, ContextBuilder.measure(messages))
        prepared, stats = ContextBuilder(ContextConfig(max_input_chars=total)).build(messages, tools=tools)
        self.assertEqual(prepared, messages)
        self.assertEqual(stats["input_chars"], total)
        with self.assertRaises(ContextLimitError) as failure:
            ContextBuilder(ContextConfig(max_input_chars=total - 1)).build(messages, tools=tools)
        self.assertEqual(failure.exception.stats["input_chars"], total)

    def test_old_turn_is_removed_as_a_whole_and_recent_order_is_preserved(self):
        old = [Message("user", "旧问题"), calculate_call(),
               Message("tool", '{"ok":true,"result":14}', tool_call_id="call_calc"),
               Message("assistant", "旧答案" * 100)]
        recent = [Message("user", "新问题"), Message("assistant", "新答案")]
        messages = [Message("system", "规则"), *old, *recent, Message("user", "当前问题")]
        snapshot = deepcopy(messages)
        expected = [messages[0], *recent, messages[-1]]
        builder = ContextBuilder(ContextConfig(max_input_chars=ContextBuilder.measure(expected)))
        prepared, stats = builder.build(messages, history_turns=[old, recent])
        self.assertEqual(prepared, expected)
        self.assertEqual(stats["history_turns_dropped"], 1)
        self.assertEqual(stats["history_turns_kept"], 1)
        self.assertFalse(any(message.role == "tool" or message.tool_calls for message in prepared))
        self.assertEqual(messages, snapshot)

    def test_memory_body_limit_priorities_deduplication_and_metadata(self):
        items = [{"id": "related", "content": "相关资料" * 100},
                 {"id": "recent", "content": "最近资料"},
                 {"id": "related", "content": "重复资料"}]
        messages = [Message("system", "规则"), Message("user", "当前问题")]
        builder = ContextBuilder(ContextConfig(max_memory_chars=30))
        prepared, stats = builder.build(messages, memory_items=items)
        self.assertIn("相关资料", prepared[1].content)
        self.assertNotIn("重复资料", prepared[1].content)
        self.assertEqual(stats["memory_ids_kept"], ["related"])
        self.assertEqual(stats["memory_ids_dropped"], ["recent"])
        self.assertEqual(stats["memory_ids_truncated"], ["related"])
        prepared, stats = ContextBuilder(ContextConfig(max_memory_chars=0)).build(messages, memory_items=items)
        self.assertEqual(prepared, messages)
        self.assertEqual(stats["memory_ids_kept"], [])

    def test_global_budget_can_remove_memory_without_truncating_current_question(self):
        messages = [Message("system", "规则"), Message("user", "完整当前问题")]
        budget = ContextBuilder.measure(messages)
        prepared, stats = ContextBuilder(ContextConfig(max_input_chars=budget)).build(
            messages, memory_items=[{"id": "m1", "content": "附加资料"}],
        )
        self.assertEqual(prepared, messages)
        self.assertEqual(stats["memory_ids_dropped"], ["m1"])

    def test_current_execution_and_system_are_mandatory_even_without_history(self):
        for messages in [
            [Message("system", "规则" * 500), Message("user", "问题")],
            [Message("system", "规则"), Message("user", "问题" * 500)],
            [Message("system", "规则"), Message("user", "原始任务"),
             Message("assistant", "当前草稿" * 500), Message("user", "评审草稿")],
            [Message("system", "规则"), Message("user", "任务"),
             calculate_call("1" * 500),
             Message("tool", '{"ok":true,"result":1}', tool_call_id="call_calc")],
        ]:
            snapshot = deepcopy(messages)
            with self.subTest(messages=len(messages)), self.assertRaises(ContextLimitError):
                ContextBuilder(ContextConfig(max_input_chars=200)).build(messages)
            self.assertEqual(messages, snapshot)

    def test_invalid_or_partial_tool_groups_never_reach_model(self):
        call = calculate_call()
        result = Message("tool", "14", tool_call_id="call_calc")
        for body in [[call], [result], [call, Message("user", "插入的问题"), result],
                     [call, result, result], [call, Message("tool", "14", tool_call_id="other")]]:
            with self.subTest(body=body), self.assertRaises(OutputError):
                ContextBuilder().build([Message("system", "规则"), *body])

    def test_multi_call_group_and_ids_are_preserved(self):
        calls = [ToolCall("a", "first", "{}"), ToolCall("b", "second", "{}")]
        messages = [Message("system", "规则"), Message("user", "任务"),
                    Message("assistant", tool_calls=calls),
                    Message("tool", "1", tool_call_id="b"), Message("tool", "2", tool_call_id="a")]
        prepared, _ = ContextBuilder().build(messages)
        self.assertEqual(prepared, messages)
        prepared[1].content = "修改发送副本"
        self.assertEqual(messages[1].content, "任务")

    def test_configuration_and_history_boundary_validation(self):
        for values in [{"max_input_chars": 0}, {"max_input_chars": True},
                       {"max_memory_chars": -1}, {"max_tool_result_chars": 127}]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                ContextConfig(**values)
        with self.assertRaises(ValueError):
            ContextBuilder().build([Message("user", "任务")])
        with self.assertRaises(ValueError):
            ContextBuilder().build([Message("system", "规则"), Message("user", "当前任务")],
                                   history_turns=[[Message("user", "历史任务")]])


class ToolPreviewTests(unittest.TestCase):
    def test_valid_json_status_and_reference_fields_survive_compaction(self):
        item = {"content": "工具注册" * 3000, "source": "/notes/guide.md", "chunk_id": "chunk1",
                "citation": "guide.md:L3-L7", "start_line": 3, "end_line": 7, "truncated": False}
        content = json.dumps({"ok": True, "result": [item]}, ensure_ascii=False)
        compact = compact_tool_result(content, 600)
        payload = json.loads(compact)
        self.assertLessEqual(len(compact), 600)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["truncated"])
        self.assertEqual(payload["original_chars"], len(content))
        self.assertEqual(payload["result"][0]["citation"], item["citation"])
        self.assertEqual(payload["result"][0]["source"], item["source"])
        self.assertEqual(payload["result"][0]["chunk_id"], item["chunk_id"])
        self.assertTrue(payload["result"][0]["truncated"])

    def test_error_status_plain_text_lists_and_nested_results_are_bounded(self):
        inputs = [
            json.dumps({"ok": False, "error": "详细错误" * 1000}, ensure_ascii=False),
            "plain result " * 1000,
            json.dumps({"ok": True, "result": list(range(1000))}),
            json.dumps({"ok": True, "result": {str(i): {"nested": ["x" * 1000]} for i in range(100)}}),
            json.dumps({"ok": True, "result": {"source": "/" + "long" * 1000}}),
        ]
        for content in inputs:
            with self.subTest(content=content[:30]):
                compact = compact_tool_result(content, 128)
                payload = json.loads(compact)
                self.assertLessEqual(len(compact), 128)
                self.assertTrue(payload["truncated"])
                if '"ok": false' in content:
                    self.assertFalse(payload["ok"])
        short = '{"ok":true,"result":14}'
        self.assertEqual(compact_tool_result(short, 128), short)


class AgentContextTests(unittest.TestCase):
    def test_raw_tool_results_remain_in_trace_history_and_sqlite(self):
        with SQLiteStore(":memory:") as db:
            tools = create_registry()
            original = "Python 文档原文" * 1000

            @tools.tool
            def long_document() -> dict:
                return {"content": original, "citation": "guide.md:L1-L1", "chunk_id": "one"}

            llm = ScriptedLLM([Message("assistant", tool_calls=[ToolCall("doc", "long_document", "{}")]),
                               "完整回答"])
            session = SessionStore(db)
            agent = ReActAgent(llm, tools=tools, session=session,
                               context=ContextBuilder(ContextConfig(max_tool_result_chars=500)))
            events = []
            agent.run("读取文档", on_event=events.append)
            sent = llm.requests[1][0][-1]
            self.assertEqual(sent["tool_call_id"], "doc")
            self.assertLessEqual(len(sent["content"]), 500)
            self.assertTrue(json.loads(sent["content"])["truncated"])
            raw = next(message for message in agent.get_history() if message.role == "tool")
            self.assertEqual(json.loads(raw.content)["result"]["content"], original)
            restored = next(message for message in session.load("ReActAgent")[0] if message.role == "tool")
            self.assertEqual(restored.content, raw.content)
            self.assertEqual(next(event for event in agent.trace if event["stage"] == "工具")
                             ["result"]["content"], original)
            stats = agent.trace[-1]["context"]
            self.assertEqual(stats["tool_results_trimmed"][0]["tool_call_id"], "doc")
            self.assertEqual(stats["tool_results_trimmed"][0]["sent_chars"], len(sent["content"]))
            self.assertTrue(any(event["type"] == "text_delta" for event in events))

    def test_history_pruning_does_not_delete_saved_turns(self):
        with SQLiteStore(":memory:") as db:
            session = SessionStore(db)
            session.append("SimpleAgent", [Message("user", "旧问题" * 1000), Message("assistant", "旧答案")])
            llm = ScriptedLLM(["新答案"])
            agent = SimpleAgent(llm, session=session, context=ContextBuilder(ContextConfig(max_input_chars=600)))
            agent.run("新问题")
            self.assertEqual(len(llm.requests[0][0]), 2)
            self.assertEqual(agent.trace[-1]["context"]["history_turns_dropped"], 1)
            self.assertEqual(len(agent.get_history()), 4)
            self.assertEqual(len(session.load("SimpleAgent")), 2)

    def test_budget_failure_skips_model_and_does_not_commit_or_leave_run_state(self):
        with SQLiteStore(":memory:") as db:
            session = SessionStore(db)
            llm = ScriptedLLM(["后续成功"])
            agent = SimpleAgent(llm, session=session, context=ContextBuilder(ContextConfig(max_input_chars=400)))
            with self.assertRaises(ContextLimitError):
                agent.run("太长的问题" * 300)
            self.assertEqual(llm.requests, [])
            self.assertEqual(session.load("SimpleAgent"), [])
            self.assertEqual(agent.trace[-1]["error"], "ContextLimitError")
            self.assertGreater(agent.trace[-1]["context"]["input_chars"], 400)
            self.assertEqual(agent._context_turns, [])
            self.assertEqual(agent._context_memories, [])
            self.assertIsNone(agent._on_event)
            self.assertEqual(agent.run("短问题"), "后续成功")

    def test_every_model_call_has_statistics_in_all_four_modes(self):
        cases = [(SimpleAgent, ["回答"], {}),
                 (ReActAgent, [calculate_call(), "14"], {"tools": create_registry()}),
                 (ReflectionAgent, ["草稿", '{"approved":false,"feedback":"修订"}', "答案",
                                    '{"approved":true,"feedback":""}'], {}),
                 (PlanAndSolveAgent, ['["计算"]', calculate_call(), "14", "答案"],
                  {"tools": create_registry()})]
        for cls, replies, options in cases:
            llm, events = ScriptedLLM(replies), []
            agent = cls(llm, **options)
            with self.subTest(mode=cls.__name__):
                agent.run("任务", on_event=events.append)
                stats = [event["context"] for event in agent.trace if "message" in event]
                self.assertEqual(len(stats), len(llm.requests))
                for context, (messages, tools) in zip(stats, llm.requests):
                    payload = {"messages": messages}
                    if tools:
                        payload["tools"] = tools
                    sent_chars = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                    self.assertEqual(context["input_chars"], sent_chars)
                    self.assertLessEqual(sent_chars, context["budget_chars"])
                starts = [event for event in events if event["type"] == "model_start"]
                self.assertEqual([event["context"] for event in starts], stats)

    def test_reflection_and_plan_internal_growth_is_checked_before_next_request(self):
        for cls, replies in [(ReflectionAgent, ["很长的草稿" * 1000]),
                             (PlanAndSolveAgent, [json.dumps(["很长的步骤" * 1000])])]:
            llm = ScriptedLLM(replies)
            agent = cls(llm, context=ContextBuilder(ContextConfig(max_input_chars=1000)))
            with self.subTest(mode=cls.__name__), self.assertRaises(ContextLimitError):
                agent.run("任务")
            self.assertEqual(len(llm.requests), 1)
            self.assertEqual(agent.get_history(), [])
            self.assertEqual(agent.trace[-1]["error"], "ContextLimitError")

    def test_memory_selection_stats_describe_sent_copy_without_changing_storage(self):
        with SQLiteStore(":memory:") as db:
            memory = MemoryStore(db)
            saved = memory.add("偏好简洁示例" * 100)
            llm = ScriptedLLM(["回答"])
            agent = SimpleAgent(llm, memory=memory,
                                context=ContextBuilder(ContextConfig(max_memory_chars=30)))
            agent.run("介绍装饰器")
            stats = agent.trace[-1]["context"]
            self.assertEqual(stats["memory_ids_kept"], [saved["id"]])
            self.assertEqual(stats["memory_ids_truncated"], [saved["id"]])
            self.assertEqual(memory.list()[0]["content"], saved["content"])
            self.assertNotIn("记忆参考", agent.get_history()[0].content)

    def test_cli_rejects_invalid_budgets_before_loading_model_or_database(self):
        with patch("sys.argv", ["agent_framework", "--tool-result-chars", "10"]), \
                patch("agent_framework.demo.LLM") as model, \
                patch("agent_framework.demo.SQLiteStore") as store, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
            main()
        self.assertEqual(failure.exception.code, 1)
        model.assert_not_called()
        store.assert_not_called()


if __name__ == "__main__":
    unittest.main()
