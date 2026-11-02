import asyncio
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ..examples.offline_demo import ScriptedLLM
from ..mini_agents import (AgentError, MCPClient, Message, PlanAndSolveAgent, ReActAgent,
                           ToolCall, ToolRegistry, create_registry, load_mcp_config)
from ..mini_agents.tools.mcp import _MCPTool, _sdk

ROOT = Path(__file__).resolve().parents[2]
HAS_MCP = importlib.util.find_spec("mcp") is not None


def text_result(result):
    return json.loads(result["content"][0]["text"])


class MCPConfigTests(unittest.TestCase):
    def test_optional_dependency_does_not_break_local_tools(self):
        with patch.dict(sys.modules, {"mcp": None}):
            with self.assertRaisesRegex(AgentError, "requirements-mcp"):
                MCPClient("python").start()
            self.assertEqual(create_registry().execute("calculator", {"expression": "2+3"})["result"], 5)

    def test_configuration_types_and_relative_cwd(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "servers.json"
            path.write_text(json.dumps({"mcpServers": {"courses": {
                "command": "python", "args": [], "cwd": "subdir",
            }}}), encoding="utf-8")
            self.assertEqual(load_mcp_config(path)["courses"]["cwd"], str((Path(directory) / "subdir").resolve()))
            for config in [[], {}, {"mcpServers": []}, {"mcpServers": {"x": {"url": "https://example.com"}}},
                           {"mcpServers": {"x": {"command": "python", "extra": 1}}}]:
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.subTest(config=config), self.assertRaises(ValueError):
                    load_mcp_config(path)
        for options in [{"command": ""}, {"command": "python", "args": "-m x"},
                        {"command": "python", "env": {"bad": 1}},
                        {"command": "python", "prefix": "bad/name"},
                        {"command": "python", "timeout": True},
                        {"command": "python", "timeout": float("nan")}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                MCPClient(**options)

    def test_tool_discovery_pagination_and_repeated_cursor(self):
        first = SimpleNamespace(model_dump=lambda **_: {"name": "first"})
        second = SimpleNamespace(model_dump=lambda **_: {"name": "second"})
        for repeat in [False, True]:
            calls = []

            async def list_tools(cursor=None):
                calls.append(cursor)
                return SimpleNamespace(tools=[second if cursor else first],
                                       nextCursor="next" if repeat or cursor is None else None)

            client = MCPClient("python")
            with patch.object(client, "_request", lambda operation: asyncio.run(
                    operation(SimpleNamespace(list_tools=list_tools)))):
                if repeat:
                    with self.assertRaisesRegex(ValueError, "重复分页"):
                        client.list_tools()
                else:
                    self.assertEqual(client.list_tools(), [{"name": "first"}, {"name": "second"}])
                self.assertEqual(calls, [None, "next"])


@unittest.skipUnless(HAS_MCP, "安装 requirements-mcp.txt 后执行真实 MCP 集成测试")
class MCPIntegrationTests(unittest.TestCase):
    def connect(self, **options):
        return self.enterContext(MCPClient(
            sys.executable, ["-m", "agent_framework.tests.mcp_fixture"],
            prefix="remote", cwd=ROOT, **options,
        ))

    def test_discovery_preserves_schema_validates_before_remote_call_and_closes_process(self):
        client = MCPClient(sys.executable, ["-m", "agent_framework.tests.mcp_fixture"],
                           cwd=ROOT, env={"MCP_TEST_MARKER": "实际子进程"}, prefix="remote")
        with client:
            definitions = client.list_tools()
            tools = create_registry()
            client.register_tools(tools, allowlist=["inspect_query"])
            schema = next(item["function"] for item in tools.schemas()
                          if item["function"]["name"] == "remote__inspect_query")
            original = next(item for item in definitions if item["name"] == "inspect_query")
            self.assertEqual(schema["parameters"], original["inputSchema"])
            self.assertEqual(schema["description"], original["description"])
            for arguments in [{"query": {"names": ["课程"], "count": "2"}},
                              {"query": {"names": ["课程"], "count": 0}},
                              {"query": {"names": ["课程"], "count": 2, "extra": True}}]:
                with self.subTest(arguments=arguments), patch.object(client, "call_tool") as remote:
                    self.assertFalse(tools.execute("remote__inspect_query", arguments)["ok"])
                    remote.assert_not_called()
            result = tools.execute("remote__inspect_query", {"query": {"names": ["课程"], "count": 2}})
            self.assertTrue(result["ok"], result)
            payload = text_result(result["result"])
            pid = payload["pid"]
            self.assertEqual(payload["marker"], "实际子进程")
            self.assertEqual(Path(payload["cwd"]).resolve(), ROOT)
            self.assertNotEqual(pid, os.getpid())
        self.assertFalse(client._thread.is_alive())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self.assertFalse(tools.execute("remote__inspect_query", {"query": {"names": [], "count": 1}})["ok"])
        client.close()

    def test_allowlist_duplicate_is_atomic_and_schema_copy(self):
        client = self.connect()
        tools = create_registry()
        with self.assertRaisesRegex(ValueError, "未找到"):
            client.register_tools(tools, allowlist=["missing"])
        self.assertEqual(len(tools.schemas()), 1)
        client.register_tools(tools, allowlist=["broken"])
        before = deepcopy(tools.schemas())
        with self.assertRaisesRegex(ValueError, "名称重复"):
            client.register_tools(tools, allowlist=["broken", "delayed"])
        self.assertEqual(before, tools.schemas())
        changed = tools.schemas()
        changed[-1]["function"]["parameters"]["properties"]["injected"] = {}
        self.assertEqual(before, tools.schemas())
        self.assertEqual(client.register_tools(tools, allowlist=[]), [])
        definitions = client.list_tools()
        definitions[0]["inputSchema"] = {"type": "invalid-type"}
        with patch.object(client, "list_tools", return_value=definitions):
            with self.assertRaisesRegex(ValueError, "inputSchema 无效"):
                client.register_tools(tools)
        self.assertEqual(before, tools.schemas())

    def test_remote_error_timeout_and_followup_request(self):
        client = self.connect(timeout=2)
        tools = ToolRegistry()
        client.register_tools(tools, allowlist=["broken", "delayed"])
        failure = tools.execute("remote__broken", {})
        self.assertFalse(failure["ok"])
        self.assertIn("示例工具失败", failure["error"])
        timeout = tools.execute("remote__delayed", {"seconds": 5})
        self.assertFalse(timeout["ok"])
        self.assertIn("超时", timeout["error"])
        self.assertTrue(tools.execute("remote__delayed", {"seconds": 0})["ok"])

    def test_missing_server_and_disconnection_cleanup(self):
        client = MCPClient("/does-not-exist/hello-agent-mcp", timeout=1)
        with self.assertRaises(AgentError):
            client.start()
        self.assertFalse(client._thread.is_alive())
        client = self.connect(timeout=2)
        tools = ToolRegistry()
        client.register_tools(tools, allowlist=["disconnect"])
        self.assertFalse(tools.execute("remote__disconnect", {})["ok"])
        client.close()
        self.assertFalse(client._thread.is_alive())

    def test_initialize_timeout_cleans_up_subprocess(self):
        with TemporaryDirectory() as directory:
            pidfile = Path(directory) / "server.pid"
            code = "import os,sys,time;open(sys.argv[1],'w').write(str(os.getpid()));time.sleep(60)"
            client = MCPClient(sys.executable, ["-c", code, str(pidfile)], timeout=1)
            with self.assertRaisesRegex(AgentError, "连接失败"):
                client.start()
            self.assertFalse(client._thread.is_alive())
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pidfile.read_text()), 0)

    def test_react_and_plan_tool_loop_preserves_events_and_context(self):
        client = self.connect()
        tools = ToolRegistry()
        client.register_tools(tools, allowlist=["inspect_query"])
        call = Message("assistant", tool_calls=[ToolCall(
            "mcp1", "remote__inspect_query", '{"query":{"names":["Agent"],"count":1}}')])
        for cls, replies in [(ReActAgent, [call, "已查询"]),
                             (PlanAndSolveAgent, ['["查询课程"]', call, "已查询", "汇总完成"])]:
            llm, events = ScriptedLLM(replies), []
            agent = cls(llm, tools=tools)
            agent.run("查询课程", on_event=events.append)
            tool_event = next(event for event in events if event["type"] == "tool_result")
            self.assertTrue(tool_event["ok"])
            self.assertEqual(text_result(tool_event["result"])["names"], ["Agent"])
            self.assertTrue(any(event["type"] == "text_delta" for event in events))
            self.assertTrue(any(json.loads(messages[-1]["content"])["ok"]
                                for messages, _ in llm.requests if messages[-1]["role"] == "tool"))
            for record in agent.trace:
                if "context" in record:
                    self.assertLessEqual(record["context"]["input_chars"], 24000)

    def test_name_mapping_handles_unusual_or_long_remote_names(self):
        client = SimpleNamespace(prefix="courses", _deps=_sdk())
        names = set()
        for remote in ["read.file", "read/file", "a" * 80, "正常中文"]:
            tool = _MCPTool(client, {"name": remote, "inputSchema": {"type": "object"}})
            self.assertRegex(tool.name, r"^[A-Za-z0-9_-]{1,64}$")
            self.assertNotIn(tool.name, names)
            names.add(tool.name)
            self.assertEqual(tool.remote_name, remote)

    def test_cli_discovery_needs_no_model_key_and_rejects_unsupported_mode(self):
        with TemporaryDirectory() as directory:
            config = Path(directory) / "servers.json"
            config.write_text(json.dumps({"mcpServers": {"courses": {
                "command": sys.executable, "args": ["-m", "agent_framework.examples.mcp_server"],
                "cwd": str(ROOT), "tools": ["get_course"],
            }}}), encoding="utf-8")
            environment = {key: value for key, value in os.environ.items()
                           if key not in {"LLM_MODEL", "LLM_MODEL_ID", "LLM_API_KEY", "OPENAI_API_KEY"}}
            command = [sys.executable, "-m", "agent_framework", "--db", ":memory:",
                       "--mcp-config", str(config)]
            result = subprocess.run([*command, "--list-tools"], cwd=ROOT, env=environment,
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            names = [tool["function"]["name"] for tool in json.loads(result.stdout)]
            self.assertIn("courses__get_course", names)
            self.assertNotIn("courses__add", names)
            invalid = subprocess.run([*command, "--mode", "simple", "任务"], cwd=ROOT,
                                     capture_output=True, text=True, timeout=15)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("只有 react 和 plan", invalid.stderr)


if __name__ == "__main__":
    unittest.main()
