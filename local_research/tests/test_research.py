import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agent_framework.examples.offline_demo import ScriptedLLM
from agent_framework.mini_agents import Config, ContextConfig, DocumentIndex, Message, ModelError, SQLiteStore, ToolCall

from local_research.__main__ import main
from local_research.report import append_sources, validate_report
from local_research.researcher import run_research
from local_research.tools import ResearchTools


def report(citations="[S1]"):
    return (f"# 示例研究\n\n## 结论摘要\n\n资料存在差异。{citations}\n\n"
            f"## 分问题分析\n\n按来源分别核对。{citations}\n\n"
            "## 对比与适用条件\n\n适用条件以资料说明为准。\n\n"
            "## 资料不足与未解决问题\n\n仍需人工核对原文是否支持结论。\n")


def call(name, arguments):
    return Message("assistant", tool_calls=[ToolCall(name + "1", name, json.dumps(arguments))])


class CorpusModel:
    """模拟模型选择；实际数据来自本轮工具消息，不伪造片段 ID。"""

    def __init__(self, final=None):
        self.requests, self.citations, self.pending = [], [], []
        self.stream_modes = []
        self.final, self.closed = final, False

    def chat(self, messages, *, tools=None, stream=True, on_token=None):
        self.requests.append(([item.to_dict() for item in messages], tools))
        self.stream_modes.append(stream)
        last = messages[-1]
        if not tools:
            text = ('["比较资料中的共享标记"]' if "将任务分解为" in last.content
                    else self.final or report("".join(f"[{item}]" for item in self.citations)))
            reply = Message("assistant", text)
        elif last.role != "tool":
            reply = call("search_documents", {"query": "共享标记"})
        else:
            payload = json.loads(last.content)
            if not payload["ok"]:
                raise ValueError("测试工具失败：" + payload["error"])
            value = payload["result"]
            if isinstance(value, list):
                self.pending.extend(item["chunk_id"] for item in value)
            else:
                self.citations.append(value["citation"])
            reply = (call("read_chunk", {"chunk_id": self.pending.pop(0)}) if self.pending
                     else Message("assistant", "已核对资料。"))
        if stream and on_token:
            for token in reply.content:
                on_token(token)
        return reply

    def close(self):
        self.closed = True


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.docs = self.root / "docs"
        self.docs.mkdir()
        self.output = self.root / "results"

    def source(self, name="guide.md", text="共享标记：方案甲耗时十分钟。"):
        path = self.docs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def run_task(self, llm=None, **options):
        return run_research("比较资料", docs=self.docs, output=self.output,
                            llm=llm if llm is not None else CorpusModel(), **options)

    def json_file(self, result, name):
        return json.loads((result.directory / name).read_text(encoding="utf-8"))

    def test_complete_loop_same_basename_conflicts_and_streaming(self):
        self.source("alpha/guide.md", "共享标记：流程A耗时十分钟。")
        self.source("beta/guide.md", "共享标记：流程A耗时二十分钟。")
        events, model = [], CorpusModel()
        result = self.run_task(model, on_event=events.append)
        self.assertEqual(result.status, "completed", result.error)
        evidence = self.json_file(result, "evidence.json")
        self.assertEqual({item["source"] for item in evidence}, {"alpha/guide.md", "beta/guide.md"})
        self.assertEqual({item["citation"] for item in evidence}, {"S1", "S2"})
        self.assertTrue(all(item["pages"] for item in evidence))
        self.assertEqual({item["pages"][0]["content"] for item in evidence},
                         {"共享标记：流程A耗时十分钟。", "共享标记：流程A耗时二十分钟。"})
        text = result.report_path.read_text(encoding="utf-8")
        self.assertIn("## 参考资料", text)
        self.assertIn("alpha/guide.md", text)
        self.assertIn("beta/guide.md", text)
        task = self.json_file(result, "task.json")
        self.assertEqual(task["plan"], ["比较资料中的共享标记"])
        self.assertLessEqual(task["model_attempts"], 20)
        self.assertEqual(task["status"], "completed")
        self.assertTrue(any(event["type"] == "text_delta" for event in events))
        self.assertEqual([message["role"] for message in model.requests[0][0]], ["system", "user", "user"])
        # 两条 user 消息分别是本次问题和框架的规划指令，不恢复其他任务历史。
        self.assertTrue((result.directory / "index.sqlite3").is_file())

    def test_empty_corpus_does_not_call_model(self):
        (self.docs / "ignored.csv").write_text("a,b\n1,2", encoding="utf-8")
        self.source("empty.txt", "   ")
        model = ScriptedLLM([])
        result = self.run_task(model)
        self.assertEqual(result.status, "insufficient")
        self.assertEqual(model.requests, [])
        self.assertIn("资料不足", result.report_path.read_text(encoding="utf-8"))

    def test_no_hits_does_not_publish_hallucinated_answer(self):
        self.source()
        llm = ScriptedLLM(['["寻找资料"]', call("search_documents", {"query": "zznomatch123"}),
                           "编造的回答[S99]", "完全编造的答案[S99]"])
        result = self.run_task(llm)
        self.assertEqual(result.status, "insufficient")
        self.assertNotIn("编造", result.report_path.read_text(encoding="utf-8"))
        self.assertIn("完全编造", (result.directory / "draft.md").read_text(encoding="utf-8"))
        self.assertEqual(self.json_file(result, "evidence.json"), [])

    def test_invalid_citation_retains_draft_and_evidence_only(self):
        self.source()
        result = self.run_task(CorpusModel(final=report("[S999]")))
        self.assertEqual(result.status, "validation_failed")
        self.assertIsNone(result.report_path)
        self.assertFalse((result.directory / "report.md").exists())
        self.assertTrue((result.directory / "draft.md").is_file())
        self.assertTrue(self.json_file(result, "evidence.json")[0]["pages"])
        self.assertIn("引用不存在", result.error)

    def test_search_preview_alone_cannot_become_a_research_report(self):
        self.source()
        model = ScriptedLLM(['["查资料"]', call("search_documents", {"query": "共享标记"}),
                             "直接引用预览[S1]", report()])
        result = self.run_task(model)
        self.assertEqual(result.status, "insufficient")
        self.assertIn("资料不足", result.report_path.read_text(encoding="utf-8"))
        self.assertEqual(self.json_file(result, "evidence.json")[0]["pages"], [])

    def test_step_limit_retains_partial_tool_trace(self):
        self.source()
        llm = ScriptedLLM(['["搜索资料"]', call("search_documents", {"query": "共享标记"})])
        result = self.run_task(llm, max_steps=1)
        self.assertEqual(result.status, "failed")
        self.assertIn("StepLimitError", result.error)
        self.assertIsNone(result.report_path)
        self.assertTrue(any(item.get("stage") == "工具" for item in self.json_file(result, "trace.json")))
        self.assertEqual(self.json_file(result, "evidence.json")[0]["pages"], [])

    def test_malformed_plan_context_limit_and_model_failure_saved(self):
        self.source()
        failures = [(ScriptedLLM(["不是计划"]), {}, "OutputError"),
                    (ScriptedLLM([]), {"context_config": ContextConfig(max_input_chars=100)}, "ContextLimitError")]
        for model, options, name in failures:
            with self.subTest(name=name):
                result = self.run_task(model, **options)
                self.assertEqual(result.status, "failed")
                self.assertIn(name, result.error)
                self.assertFalse((result.directory / "report.md").exists())
                self.assertTrue(self.json_file(result, "trace.json"))
        for failure, status in [(ModelError("模拟网络错误"), "failed"), (KeyboardInterrupt(), "interrupted")]:
            class BrokenModel:
                def chat(self, *args, **kwargs):
                    raise failure
            result = self.run_task(BrokenModel())
            self.assertEqual(result.status, status)
            self.assertIsNotNone(self.json_file(result, "task.json")["finished_at"])

    def test_invalid_utf8_preserves_failed_task(self):
        self.source()
        (self.docs / "bad.txt").write_bytes(b"\xff\xfe\xff")
        result = self.run_task(ScriptedLLM([]))
        self.assertEqual(result.status, "failed")
        self.assertIn("UnicodeDecodeError", result.error)
        for name in ("task.json", "trace.json", "evidence.json"):
            self.assertTrue((result.directory / name).exists())

    def test_new_run_does_not_use_removed_files_or_generated_reports(self):
        self.source("a.md")
        old = self.source("b.md", "共享标记：即将删除的资料。")
        first = self.run_task()
        old.unlink()
        second = self.run_task()
        self.assertNotEqual(first.task_id, second.task_id)
        self.assertEqual([item["source"] for item in self.json_file(second, "evidence.json")], ["a.md"])
        self.assertEqual(len(self.json_file(first, "evidence.json")), 2)
        nested_output = self.docs / "generated"
        prior = nested_output / "prior"
        prior.mkdir(parents=True)
        (prior / "report.md").write_text("共享标记：不能导入旧报告。", encoding="utf-8")
        outside = self.root / "outside.md"
        outside.write_text("共享标记：目录外资料。", encoding="utf-8")
        (self.docs / "outside.md").symlink_to(outside)
        result = run_research("比较资料", docs=self.docs, output=nested_output, llm=CorpusModel())
        self.assertEqual([item["source"] for item in self.json_file(result, "task.json")["imports"]], ["a.md"])

    def test_invalid_arguments_fail_before_task_creation(self):
        for options in [{"max_steps": 0}, {"max_plan_steps": 4}, {"max_steps": True}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_task(**options)
        with self.assertRaises(ValueError):
            run_research(" ", docs=self.docs, output=self.output, llm=ScriptedLLM([]))
        with self.assertRaises(ValueError):
            run_research("问题", docs=self.docs, output=self.root, llm=ScriptedLLM([]))
        self.assertFalse(self.output.exists())

    def test_cli_streaming_and_no_stream_close_model_and_return_status(self):
        self.source()
        for flags, expected in [([], "[规划]"), (["--no-stream"], "## 参考资料")]:
            model, output = CorpusModel(), io.StringIO()
            args = ["local_research", "比较资料", "--docs", str(self.docs), "--output", str(self.output), *flags]
            with patch("sys.argv", args), patch("local_research.__main__.load_dotenv"), \
                 patch("local_research.__main__.Config.from_env", return_value=Config(model="test", api_key="dummy")), \
                 patch("local_research.__main__.LLM", return_value=model), contextlib.redirect_stdout(output):
                self.assertEqual(main(), 0)
            self.assertTrue(model.closed)
            self.assertTrue(all(mode is (not bool(flags)) for mode in model.stream_modes))
            self.assertIn(expected, output.getvalue())
            if flags:
                self.assertNotIn("[规划]", output.getvalue())


class EvidenceTests(unittest.TestCase):
    def test_pagination_ids_snapshot_and_validation_before_execution(self):
        with TemporaryDirectory() as directory, SQLiteStore(":memory:") as db:
            root = Path(directory)
            source = root / "long.md"
            original = "共享标记：甲乙丙丁；" * 160
            source.write_text(original, encoding="utf-8")
            index = DocumentIndex(db)
            index.add_file(source, chunk_size=2000, overlap=0)
            tools = ResearchTools(index, root)
            self.assertFalse(tools.registry.execute("read_chunk", {"chunk_id": "missing"})["ok"])
            hit = tools.search_documents("共享标记")[0]
            self.assertEqual(hit["citation"], tools.search_documents("共享标记")[0]["citation"])
            self.assertEqual(len(hit["content"]), 200)
            self.assertEqual(hit["next_offset"], 200)
            tools.read_chunk(hit["chunk_id"], offset=hit["total_chars"])
            self.assertEqual(tools.snapshot()[0]["pages"], [])  # 空的 EOF 页不构成已读证据。
            source.write_text("已修改，不应影响索引快照", encoding="utf-8")
            content, offset = "", 0
            while offset is not None:
                page = tools.read_chunk(hit["chunk_id"], offset=offset, max_chars=123)
                content += page["content"]
                offset = page["next_offset"]
            self.assertEqual(content, original)
            snapshot = tools.snapshot()
            snapshot[0]["pages"].clear()
            self.assertTrue(tools.snapshot()[0]["pages"])
            before = tools.snapshot()
            for values in [{"offset": "0"}, {"offset": -1}, {"max_chars": 0}, {"max_chars": True}, {"extra": 1}]:
                self.assertFalse(tools.registry.execute("read_chunk", {"chunk_id": hit["chunk_id"], **values})["ok"])
            self.assertEqual(before, tools.snapshot())

    def test_citation_validation_and_program_generated_references(self):
        evidence = [{"citation": "S1", "source": "a/guide.md", "start_line": 2, "end_line": 5,
                     "pages": [{"content": "真实读取内容"}]}, {"citation": "S2", "pages": []}]
        errors, used = validate_report(report(), evidence)
        self.assertEqual(errors, [])
        self.assertEqual(used, ["S1"])
        self.assertIn("`a/guide.md`，L2–L5", append_sources(report(), evidence, used))
        for citation in ["[S999]", "[S2]", "[S0]", "[s1]", "[S1,S2]", "`[S1]`", "[S1](https://example.com)"]:
            with self.subTest(citation=citation):
                self.assertTrue(validate_report(report(citation), evidence)[0])
        self.assertTrue(validate_report(report() + "\n## 参考资料\n伪造来源", evidence)[0])


if __name__ == "__main__":
    unittest.main()
