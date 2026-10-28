"""任务笔记和分段阅读：重启恢复、隔离、游标正确性，以及模型实际接收到的内容。"""

import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ..demo import local_command, main
from ..examples.offline_demo import ScriptedLLM
from ..mini_agents import (
    ContextBuilder, ContextConfig, DocumentIndex, MemoryStore, Message, NoteStore,
    PlanAndSolveAgent, ReActAgent, ReflectionAgent, SessionStore, SimpleAgent, SQLiteStore,
    ToolCall, ToolRegistry, register_knowledge_tools,
)
from ..mini_agents.core.context import compact_tool_result


class NotesAndReadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "agent.db"
        self.db = SQLiteStore(self.path)
        self.addCleanup(self.db.close)
        self.notes = NoteStore(self.db, user_id="a", session_id="learning")
        self.documents = DocumentIndex(self.db, namespace="a")

    def document(self, content=None):
        path = self.root / "guide.md"
        text = content if content is not None else "Python 工具介绍。" * 40 + "课程代号：K7429。"
        path.write_text(text, encoding="utf-8")
        self.documents.add_file(path, chunk_size=2000, overlap=0)
        item = self.documents.search("Python")[0]
        return path, text.strip(), item["chunk_id"]

    def test_note_snapshot_replaces_previous_and_survives_reopen(self):
        self.assertIsNone(self.notes.get())
        self.notes.save("目标：学习 Agent\n下一步：工具注册")
        saved = self.notes.save("目标：学习 Agent\n已完成：工具注册\n下一步：RAG")
        with SQLiteStore(self.path) as reopened:
            notes = NoteStore(reopened, user_id="a", session_id="learning")
            self.assertEqual(notes.get(), saved)
            with reopened.transaction() as connection:
                self.assertEqual(connection.execute("SELECT count(*) FROM task_notes").fetchone()[0], 1)
        changed = self.notes.get()
        changed["content"] = "未保存的改动"
        self.assertEqual(self.notes.get()["content"], saved["content"])
        for content in ("", "  ", "x" * 2001, None):
            with self.subTest(content=str(content)[:10]), self.assertRaises(ValueError):
                self.notes.save(content)
        self.assertEqual(self.notes.get(), saved)

    def test_notes_isolated_by_user_and_session(self):
        self.notes.save("a 的主任务")
        other_session = NoteStore(self.db, user_id="a", session_id="another")
        other_user = NoteStore(self.db, user_id="b", session_id="learning")
        self.assertIsNone(other_session.get())
        self.assertIsNone(other_user.get())
        other_session.save("a 的另一任务")
        other_user.save("b 的任务")
        self.assertTrue(other_session.clear())
        self.assertFalse(other_session.clear())
        self.assertEqual(self.notes.get()["content"], "a 的主任务")
        self.assertEqual(other_user.get()["content"], "b 的任务")

    def test_all_modes_use_note_without_saving_it_in_chat_or_long_term_memory(self):
        self.notes.save("当前任务：完成分页阅读；下一步：检查 next_offset")
        memory = MemoryStore(self.db, user_id="a")
        cases = [(SimpleAgent, ["回答"]), (ReActAgent, ["回答"]),
                 (ReflectionAgent, ["草稿", '{"approved":true,"feedback":""}']),
                 (PlanAndSolveAgent, ['["说明"]', "步骤答案", "答案"])]
        for cls, replies in cases:
            llm = ScriptedLLM(replies)
            session = SessionStore(self.db, user_id="a", session_id="learning")
            agent = cls(llm, notes=self.notes, memory=memory, session=session)
            with self.subTest(mode=cls.__name__):
                events = []
                agent.run("继续任务", on_event=events.append)
                self.assertTrue(all("分页阅读" in request[0][1]["content"] for request in llm.requests))
                self.assertTrue(all(event["context"]["note_included"] for event in agent.trace
                                    if "message" in event))
                self.assertEqual([message.role for message in agent.get_history()], ["user", "assistant"])
                self.assertEqual(len(session.load(cls.__name__)[0]), 2)
                self.assertTrue(any(event["type"] == "text_delta" for event in events))
                agent.clear_history()
                self.assertIsNotNone(self.notes.get())
                self.assertIsNone(agent._context_note)
        self.assertEqual(memory.list(), [])

    def test_note_respects_total_input_budget_without_deleting_saved_data(self):
        note = self.notes.save("当前任务" * 100)
        messages = [Message("system", "规则"), Message("user", "当前问题")]
        builder = ContextBuilder(ContextConfig(max_input_chars=ContextBuilder.measure(messages)))
        prepared, stats = builder.build(messages, note=note)
        self.assertEqual(prepared, messages)
        self.assertTrue(stats["note_present"])
        self.assertFalse(stats["note_included"])
        self.assertEqual(stats["note_chars"], 0)
        self.assertEqual(self.notes.get(), note)

    def test_note_cli_commands_need_no_model_and_clear_history_keeps_note(self):
        for command in ("/note 目标：学习分页阅读", "/notes", "/clear"):
            with self.subTest(command=command), patch("sys.argv", [
                "agent_framework", "--db", str(self.path), "--user", "a", "--session", "learning", command,
            ]), patch("agent_framework.demo.Config.from_env", side_effect=AssertionError("不应读取模型配置")), \
                    patch("agent_framework.demo.LLM") as model, contextlib.redirect_stdout(io.StringIO()):
                main()
                model.assert_not_called()
        self.assertIn("分页阅读", self.notes.get()["content"])
        with contextlib.redirect_stdout(io.StringIO()):
            local_command("/note-clear", notes=self.notes, memory=MemoryStore(self.db), clear_history=lambda: None)
        self.assertIsNone(self.notes.get())

    def test_search_tool_returns_preview_and_read_chunk_returns_exact_text(self):
        _, text, chunk_id = self.document()
        tools = register_knowledge_tools(ToolRegistry(), documents=self.documents)
        hit = tools.execute("search_documents", {"query": "Python"})["result"][0]
        self.assertEqual(hit["content"], text[:200])
        self.assertEqual(hit["next_offset"], 200)
        self.assertTrue(hit["truncated"])
        self.assertNotIn("K7429", hit["content"])
        page = tools.execute("read_chunk", {"chunk_id": chunk_id, "offset": 200})["result"]
        self.assertEqual(page["content"], text[200:])
        self.assertIn("K7429", page["content"])
        self.assertIsNone(page["next_offset"])
        self.assertEqual(page["citation"], hit["citation"])
        self.assertEqual(self.documents.search("Python")[0]["content"], text)

    def test_pagination_reconstructs_unicode_and_handles_end_of_chunk(self):
        _, text, chunk_id = self.document()
        parts, offset = [], 0
        while offset is not None:
            page = self.documents.read_chunk(chunk_id, offset=offset, max_chars=37)
            self.assertEqual(page["offset"], offset)
            parts.append(page["content"])
            offset = page["next_offset"]
        self.assertEqual("".join(parts), text)
        end = self.documents.read_chunk(chunk_id, offset=len(text), max_chars=1)
        self.assertEqual(end["content"], "")
        self.assertIsNone(end["next_offset"])

    def test_budget_compaction_updates_cursor_and_never_skips_unseen_text(self):
        _, text, chunk_id = self.document()
        offset, parts, trimmed = 0, [], False
        while offset is not None:
            original = self.documents.read_chunk(chunk_id, offset=offset, max_chars=300)
            content = json.dumps({"ok": True, "result": original}, ensure_ascii=False)
            shortened = compact_tool_result(content, 500)
            self.assertLessEqual(len(shortened), 500)
            page = json.loads(shortened)["result"]
            self.assertIsInstance(page, dict)
            self.assertEqual(page["content"], text[offset:offset + len(page["content"])])
            self.assertEqual(page["citation"], original["citation"])
            self.assertGreater(len(page["content"]), 0)
            next_offset = offset + len(page["content"])
            self.assertEqual(page["next_offset"], next_offset if next_offset < len(text) else None)
            trimmed = trimmed or len(page["content"]) < len(original["content"])
            parts.append(page["content"])
            offset = page["next_offset"]
        self.assertTrue(trimmed)
        self.assertEqual("".join(parts), text)
        # 搜索返回的片段列表在被进一步缩短时，也要同步各条预览的游标。
        tools = register_knowledge_tools(ToolRegistry(), documents=self.documents)
        result = tools.execute("search_documents", {"query": "Python"})
        result = json.loads(compact_tool_result(json.dumps(result, ensure_ascii=False), 500))
        preview = result["result"][0]
        self.assertEqual(preview["next_offset"], len(preview["content"]))

    def test_reading_is_scoped_validated_and_based_on_index_snapshot(self):
        path, text, chunk_id = self.document()
        other = DocumentIndex(self.db, namespace="b")
        with self.assertRaises(ValueError):
            other.read_chunk(chunk_id)
        for options in [{"offset": -1}, {"offset": True}, {"offset": len(text) + 1},
                        {"max_chars": 0}, {"max_chars": 2001}, {"max_chars": False}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.documents.read_chunk(chunk_id, **options)
        tools = register_knowledge_tools(ToolRegistry(), documents=self.documents)
        self.assertFalse(tools.execute("read_chunk", {"chunk_id": chunk_id, "namespace": "b"})["ok"])
        self.assertFalse(tools.execute("read_chunk", {"chunk_id": "unknown"})["ok"])
        path.write_text("Python 更新后的资料", encoding="utf-8")
        self.assertEqual(self.documents.read_chunk(chunk_id)["content"], text)
        self.documents.add_file(path)
        with self.assertRaises(ValueError):
            self.documents.read_chunk(chunk_id)

    def test_react_and_plan_can_search_then_read_before_answering(self):
        _, _, chunk_id = self.document()
        calls = [Message("assistant", tool_calls=[ToolCall("search", "search_documents", '{"query":"Python"}')]),
                 Message("assistant", tool_calls=[ToolCall("read", "read_chunk", json.dumps({
                     "chunk_id": chunk_id, "offset": 200,
                 }))])]
        for cls, replies in [(ReActAgent, [*calls, "课程代号 K7429"]),
                             (PlanAndSolveAgent, ['["查询课程代号"]', *calls, "K7429", "课程代号 K7429"])]:
            llm = ScriptedLLM(replies)
            tools = register_knowledge_tools(ToolRegistry(), documents=self.documents)
            agent = cls(llm, tools=tools)
            with self.subTest(mode=cls.__name__):
                self.assertIn("K7429", agent.run("查询课程代号"))
                results = [event for event in agent.trace if event["stage"] == "工具"]
                self.assertNotIn("K7429", results[0]["result"][0]["content"])
                self.assertIn("K7429", results[1]["result"]["content"])
                self.assertTrue(all(event["context"]["input_chars"] <= event["context"]["budget_chars"]
                                    for event in agent.trace if "message" in event))


if __name__ == "__main__":
    unittest.main()
