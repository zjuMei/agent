"""使用临时数据库和真实文本验证持久化、隔离、检索与 Agent 闭环，不联网。"""

import asyncio
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ..demo import local_command, main
from ..examples.offline_demo import ScriptedLLM, calculate_call
from ..mini_agents import (
    AgentError, AsyncToolExecutor, DocumentIndex, MemoryStore, Message, OutputError,
    PlanAndSolveAgent, ReActAgent, ReflectionAgent, SessionStore, SimpleAgent, SQLiteStore,
    ToolCall, create_registry, register_knowledge_tools,
)
from ..mini_agents.retrieval import split_document


class LocalTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db_path = self.root / "data" / "agent.db"
        self.db = SQLiteStore(self.db_path)
        self.addCleanup(self.db.close)

    def session(self, user="a", name="chat"):
        return SessionStore(self.db, user_id=user, session_id=name)

    def memory(self, user="a"):
        return MemoryStore(self.db, user_id=user)

    def documents(self, user="a"):
        return DocumentIndex(self.db, namespace=user)

    def file(self, text, name="guide.md"):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_reopen_restores_complete_tool_protocol_and_streaming(self):
        first = ReActAgent(ScriptedLLM([calculate_call(), "14"]), tools=create_registry(),
                           session=self.session())
        first.run("计算表达式")
        with SQLiteStore(self.db_path) as reopened:
            llm = ScriptedLLM(["仍然是14"])
            agent = ReActAgent(llm, tools=create_registry(),
                               session=SessionStore(reopened, user_id="a", session_id="chat"))
            self.assertEqual([m.to_dict() for m in agent.get_history()],
                             [m.to_dict() for m in first.get_history()])
            events = []
            agent.run("刚才的结果呢？", on_event=events.append)
            messages = llm.requests[0][0]
            self.assertEqual(messages[2]["tool_calls"][0]["id"], messages[3]["tool_call_id"])
            self.assertEqual(json.loads(messages[3]["content"])["result"], 14)
            self.assertTrue(any(event["type"] == "text_delta" for event in events))

    def test_disk_keeps_all_successes_but_context_loads_recent_complete_turns(self):
        llm = ScriptedLLM(["一", "二", "三"])
        agent = SimpleAgent(llm, session=self.session(), max_turns=1)
        for query in ("第一问", "第二问", "第三问"):
            agent.run(query)
        self.assertEqual([m.content for m in agent.get_history()], ["第三问", "三"])
        self.assertEqual([m["content"] for m in llm.requests[-1][0]],
                         [agent.system_prompt, "第二问", "二", "第三问"])
        self.assertEqual(len(self.session().load("SimpleAgent", limit=10)), 3)

    def test_failed_response_and_failed_database_write_do_not_commit_turns(self):
        session = self.session()
        agent = SimpleAgent(ScriptedLLM(["成功", " ", "写入失败"]), session=session)
        agent.run("第一问")
        with self.assertRaises(OutputError):
            agent.run("第二问")
        with patch.object(session, "append", side_effect=AgentError("模拟磁盘写入失败")):
            with self.assertRaises(AgentError):
                agent.run("第三问")
        self.assertEqual(len(session.load("SimpleAgent")), 1)
        self.assertEqual([m.content for m in agent.get_history()], ["第一问", "成功"])

    def test_sessions_modes_users_and_clear_are_isolated(self):
        cases = [(SimpleAgent, self.session(), ["主会话"]),
                 (SimpleAgent, self.session(name="other"), ["其他会话"]),
                 (SimpleAgent, self.session(user="b"), ["其他用户"]),
                 (ReflectionAgent, self.session(), ["草稿", '{"approved":true,"feedback":""}'])]
        agents = []
        for cls, session, replies in cases:
            agent = cls(ScriptedLLM(replies), session=session)
            agent.run("问题")
            agents.append(agent)
        memory = self.memory()
        memory.add("需要保留的偏好")
        documents = self.documents()
        documents.add_file(self.file("需要保留的文档"))
        agents[0].clear_history()
        self.assertEqual(self.session().load("SimpleAgent"), [])
        for cls, session, _ in cases[1:]:
            self.assertEqual(len(session.load(cls.__name__)), 1)
        self.assertEqual(len(memory.list()), 1)
        self.assertEqual(len(documents.search("文档")), 1)

    def test_corrupted_session_reports_error_instead_of_replaying_bad_messages(self):
        with self.db.transaction() as connection:
            connection.execute("INSERT INTO turns(user_id,session_id,agent_type,messages) "
                               "VALUES ('a','chat','SimpleAgent','{}')")
        with self.assertRaisesRegex(AgentError, "损坏"):
            SimpleAgent(ScriptedLLM([]), session=self.session())

    def test_memory_crud_dedup_validation_and_user_scope(self):
        memory = self.memory()
        item = memory.add("我喜欢简洁的 Python 示例")
        self.assertEqual(memory.add("  我喜欢简洁的 Python 示例  ")["id"], item["id"])
        self.assertEqual(len(memory.list()), 1)
        self.assertEqual(memory.search("Python 示例")[0]["id"], item["id"])
        self.assertFalse(self.memory("b").update(item["id"], "越权修改"))
        self.assertFalse(self.memory("b").delete(item["id"]))
        self.assertEqual(self.memory("b").search("Python"), [])
        self.assertTrue(memory.update(item["id"], "我正在学习 Agent"))
        self.assertEqual(memory.search("Python"), [])
        for content in ("", "  ", "x" * 2001):
            with self.subTest(content=content[:10]), self.assertRaises(ValueError):
                memory.add(content)
        self.assertTrue(memory.delete(item["id"]))
        self.assertEqual(memory.list(), [])

    def test_memory_context_is_bounded_and_never_saved_as_conversation(self):
        memory = self.memory()
        memory.add("我喜欢简洁示例")
        memory.add("Python " * 200)
        context = memory.context("Python", max_chars=30)
        self.assertLessEqual(sum(len(item["content"]) for item in context), 30)
        self.assertTrue(any(item["truncated"] for item in context))
        llm = ScriptedLLM(["回答"])
        agent = SimpleAgent(llm, session=self.session(), memory=memory)
        agent.run("装饰器是什么？")
        self.assertIn("我喜欢简洁示例", llm.requests[0][0][1]["content"])
        self.assertEqual([m.content for m in agent.get_history()], ["装饰器是什么？", "回答"])
        self.assertEqual(len(self.session().load("SimpleAgent")[0]), 2)
        self.assertEqual(len(memory.list()), 2)  # 回答不会自动成为长期记忆。
        memory.delete(memory.list()[0]["id"])
        memory.delete(memory.list()[0]["id"])
        agent = SimpleAgent(ScriptedLLM(["新回答"]), session=self.session(), memory=memory)
        agent.run("继续")
        self.assertFalse(any(event["stage"] == "记忆" for event in agent.trace))

    def test_memory_context_available_to_all_four_modes(self):
        memory = self.memory()
        memory.add("偏好简洁示例")
        cases = [(SimpleAgent, ["回答"]), (ReActAgent, ["回答"]),
                 (ReflectionAgent, ["回答", '{"approved":true,"feedback":""}']),
                 (PlanAndSolveAgent, ['["介绍"]', "步骤回答", "回答"])]
        for cls, replies in cases:
            llm = ScriptedLLM(replies)
            with self.subTest(mode=cls.__name__):
                cls(llm, memory=memory).run("介绍装饰器")
                self.assertTrue(all("偏好简洁示例" in request[0][1]["content"]
                                    for request in llm.requests))

    def test_chunks_cover_long_text_with_overlap_and_real_line_numbers(self):
        text = "# 标题\n\n第一段介绍注册工具。\n\n" + "长段落没有换行" * 30 + "\n结尾。"
        chunks = split_document(text, chunk_size=60, overlap=10)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(0 < len(chunk["content"]) <= 60 for chunk in chunks))
        self.assertIn("第一段介绍注册工具。", "".join(chunk["content"] for chunk in chunks))
        for chunk in chunks:
            lines = text.splitlines()[chunk["start_line"] - 1:chunk["end_line"]]
            self.assertIn(chunk["content"], "\n".join(lines))
        windows = split_document("abcdefghijklmno", chunk_size=6, overlap=2)
        self.assertEqual([item["content"] for item in windows], ["abcdef", "efghij", "ijklmn", "mno"])
        self.assertEqual(split_document("\n \n"), [])
        for size, overlap in [(0, 0), (10, 10), (10, -1)]:
            with self.assertRaises(ValueError):
                split_document(text, chunk_size=size, overlap=overlap)

    def test_document_import_search_sources_budget_no_hit_and_namespace(self):
        documents = self.documents()
        file = self.file("# 教程\n\nPython 工具注册使用装饰器。\n\n课程时长是45分钟。\n")
        documents.add_file(file)
        result = documents.search("Python 工具注册")
        self.assertIn("装饰器", result[0]["content"])
        self.assertEqual(result[0]["source"], str(file.resolve()))
        self.assertEqual(result[0]["citation"], "guide.md:L1-L5")
        self.assertIn("工具注册", documents.search("工具注册")[0]["content"])
        self.assertEqual(documents.search("astronomy"), [])
        self.assertEqual(self.documents("b").search("Python"), [])
        result = documents.search("Python", max_chars=5)
        self.assertEqual(len(result[0]["content"]), 5)
        self.assertTrue(result[0]["truncated"])
        for options in [{"limit": 0}, {"limit": True}, {"max_chars": 0}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                documents.search("Python", **options)

    def test_reimport_replaces_old_chunks_and_remove_is_scoped(self):
        file = self.file("Python decorators")
        documents = self.documents()
        documents.add_file(file)
        old_id = documents.search("Python")[0]["chunk_id"]
        documents.add_file(file)
        self.assertEqual(len(documents.search("Python")), 1)
        self.assertEqual(documents.search("Python")[0]["chunk_id"], old_id)
        self.documents("b").add_file(file)
        file.write_text("SQLite transactions", encoding="utf-8")
        documents.add_file(file)
        self.assertEqual(documents.search("Python"), [])
        self.assertEqual(len(documents.search("SQLite")), 1)
        self.assertEqual(self.documents("b").remove(file), 1)
        self.assertEqual(len(documents.search("SQLite")), 1)
        file.write_text("", encoding="utf-8")
        documents.add_file(file)
        self.assertEqual(documents.search("SQLite"), [])

    def test_import_failure_preserves_previous_index_and_directory_skips_external_symlink(self):
        file = self.file("Python original")
        documents = self.documents()
        documents.add_file(file)
        file.write_bytes(b"\xff\xfe")
        with self.assertRaises(UnicodeError):
            documents.add_file(file)
        self.assertEqual(len(documents.search("Python")), 1)
        knowledge = self.root / "knowledge"
        knowledge.mkdir()
        (knowledge / "one.txt").write_text("Python one", encoding="utf-8")
        (knowledge / "two.md").write_text("Python two", encoding="utf-8")
        (knowledge / "ignored.csv").write_text("Python ignored", encoding="utf-8")
        (knowledge / "outside.md").symlink_to(file)
        self.assertEqual(len(documents.add_directory(knowledge)), 2)
        with self.assertRaises(ValueError):
            documents.add_file(knowledge / "ignored.csv")

    def test_search_tools_are_read_only_scoped_and_work_with_thread_executor(self):
        memory, documents = self.memory(), self.documents()
        memory.add("学习 Python")
        documents.add_file(self.file("Python 装饰器用于注册工具。"))
        tools = register_knowledge_tools(create_registry(), memory=memory, documents=documents)
        schemas = {item["function"]["name"]: item["function"] for item in tools.schemas()}
        self.assertEqual(set(schemas), {"calculator", "search_memory", "search_documents", "read_chunk"})
        for name in ("search_memory", "search_documents"):
            self.assertEqual(set(schemas[name]["parameters"]["properties"]), {"query"})
            self.assertFalse(tools.execute(name, {"query": "Python", "user_id": "b"})["ok"])
        results = asyncio.run(AsyncToolExecutor(tools).run([
            ("search_memory", {"query": "Python"}), ("search_documents", {"query": "Python"}),
        ]))
        self.assertTrue(all(result["ok"] and result["result"] for result in results))

    def test_react_and_plan_retrieve_real_document_before_answering(self):
        documents = self.documents()
        documents.add_file(self.file("课程时长是45分钟。"))
        call = Message("assistant", tool_calls=[
            ToolCall("docs1", "search_documents", json.dumps({"query": "课程时长"})),
        ])
        for cls, replies in [(ReActAgent, [call, "45分钟 [guide.md:L1-L1]"]),
                             (PlanAndSolveAgent, ['["查询课程"]', call, "45分钟", "45分钟 [guide.md:L1-L1]"])]:
            llm, events = ScriptedLLM(replies), []
            tools = register_knowledge_tools(create_registry(), documents=documents)
            agent = cls(llm, tools=tools, session=self.session())
            with self.subTest(mode=cls.__name__):
                self.assertIn("45分钟", agent.run("查课程时长", on_event=events.append))
                tool_event = next(event for event in events if event["type"] == "tool_result")
                self.assertEqual(tool_event["result"][0]["citation"], "guide.md:L1-L1")
                self.assertTrue(any(event["type"] == "text_delta" for event in events))
                self.assertEqual(len(self.session().load(cls.__name__)), 1)

    def test_cli_memory_commands_do_not_call_model_or_change_chat(self):
        memory, agent = self.memory(), SimpleAgent(ScriptedLLM([]), session=self.session())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(local_command("/remember 学习 Python", memory=memory,
                                         clear_history=agent.clear_history))
            item = memory.list()[0]
            local_command(f"/update {item['id']} 学习 Agent", memory=memory,
                          clear_history=agent.clear_history)
            self.assertEqual(memory.list()[0]["content"], "学习 Agent")
            local_command("/clear", memory=memory, clear_history=agent.clear_history)
            self.assertEqual(len(memory.list()), 1)
            local_command(f"/forget {item['id']}", memory=memory, clear_history=agent.clear_history)
        self.assertEqual(memory.list(), [])
        self.assertEqual(agent.llm.requests, [])

    def test_cli_import_and_maintenance_work_without_model_configuration(self):
        file = self.file("Python 入门")
        for arguments in [
            ["--ingest", str(file), "--ingest-only"], ["/remember 我正在学习 Python"], ["/clear"],
        ]:
            with self.subTest(arguments=arguments), patch("sys.argv", [
                "agent_framework", "--db", str(self.db_path), "--user", "a", *arguments,
            ]), patch("agent_framework.demo.LLM") as model, \
                    patch("agent_framework.demo.Config.from_env", side_effect=AssertionError("不应读取模型配置")), \
                    contextlib.redirect_stdout(io.StringIO()):
                main()
                model.assert_not_called()
        self.assertEqual(len(self.documents().search("Python")), 1)
        self.assertEqual(len(self.memory().list()), 1)


if __name__ == "__main__":
    unittest.main()
