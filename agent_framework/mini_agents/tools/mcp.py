"""可选的 stdio MCP 客户端：复用服务端 Schema，适配同步 ToolRegistry。"""

import asyncio
from concurrent.futures import Future, TimeoutError as FutureTimeout
from copy import deepcopy
from datetime import timedelta
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from threading import RLock, Thread

from ..core.exceptions import AgentError
from .base import Tool


def _sdk():
    # 延迟导入：没有安装 MCP 时，普通 Agent 和本地工具仍可使用。
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from jsonschema.validators import validator_for
        from referencing import Registry
    except ImportError as error:
        raise AgentError("MCP 需要可选依赖，请运行："
                         "python -m pip install -r agent_framework/requirements-mcp.txt") from error
    return ClientSession, StdioServerParameters, stdio_client, validator_for, Registry


def _error_text(error):
    # AnyIO 可能用 ExceptionGroup 包装启动错误，保留真正的异常原因。
    nested = getattr(error, "exceptions", None)
    return "; ".join(map(_error_text, nested)) if nested else f"{type(error).__name__}: {error}"


class MCPClient:
    """一个实例管理一个本地服务。使用 with；不自动重试或重新执行工具。"""

    def __init__(self, command: str, args=None, *, prefix="mcp", timeout=30, cwd=None, env=None):
        if not isinstance(command, str) or not command.strip():
            raise ValueError("MCP command 必须是非空可执行文件名，参数另放 args")
        if args is not None and (not isinstance(args, list) or
                                 any(not isinstance(arg, str) for arg in args)):
            raise ValueError("MCP args 必须是字符串列表")
        if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,24}", prefix):
            raise ValueError("MCP 服务名只能含字母、数字、下划线和连字符，最长 24 字符")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or
                not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("MCP timeout 必须是正数秒")
        if cwd is not None and not isinstance(cwd, (str, Path)):
            raise ValueError("MCP cwd 必须是目录路径")
        if env is not None and (not isinstance(env, dict) or
                                any(not isinstance(k, str) or not isinstance(v, str)
                                    for k, v in env.items())):
            raise ValueError("MCP env 必须是字符串到字符串的映射")
        self.command, self.args = command, list(args or [])
        self.prefix, self.timeout = prefix, float(timeout)
        self.cwd, self.env = cwd, deepcopy(env)
        self._lock = RLock()
        self._ready = Future()
        self._thread = self._loop = self._session = self._stop = self._owner = None
        self._closed = False
        self._failure = None

    def start(self):
        with self._lock:
            if self._closed:
                raise AgentError("MCP 客户端已关闭，请创建新实例")
            if self._thread is not None:
                if self._session is None:
                    raise AgentError("MCP 连接不可用")
                return self
            self._deps = _sdk()
            self._thread = Thread(target=self._worker, name=f"mcp-{self.prefix}", daemon=True)
            self._thread.start()
            try:
                self._ready.result(timeout=self.timeout + 5)
            except FutureTimeout as error:
                self.close()
                raise AgentError(f"MCP {self.prefix} 启动超时") from error
            except BaseException:
                self.close()
                raise
            return self

    def _worker(self):
        try:
            asyncio.run(self._serve())
        except BaseException as error:
            self._failure = _error_text(error)
            if not self._ready.done():
                self._ready.set_exception(AgentError(f"MCP {self.prefix} 连接失败：{self._failure}"))
        finally:
            self._session = None

    async def _serve(self):
        ClientSession, Parameters, stdio_client, _, _ = self._deps
        self._loop, self._owner = asyncio.get_running_loop(), asyncio.current_task()
        self._stop = asyncio.Event()
        parameters = Parameters(command=self.command, args=self.args, cwd=self.cwd, env=self.env)
        # stdio 和 Session 的进入、退出始终位于同一个任务，满足 SDK 取消作用域要求。
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer, read_timeout_seconds=timedelta(seconds=self.timeout)) as session:
                await asyncio.wait_for(session.initialize(), timeout=self.timeout)
                self._session = session
                self._ready.set_result(None)
                await self._stop.wait()

    def _request(self, operation):
        with self._lock:
            if self._closed or self._session is None or not self._thread.is_alive():
                raise AgentError("MCP 连接未开启或已关闭" + (f"：{self._failure}" if self._failure else ""))

            async def invoke():
                return await asyncio.wait_for(operation(self._session), timeout=self.timeout)

            request = invoke()
            try:
                pending = asyncio.run_coroutine_threadsafe(request, self._loop)
            except RuntimeError as error:
                request.close()  # 服务恰好在提交时退出，避免遗留未 await 的协程。
                raise AgentError(f"MCP {self.prefix} 连接已关闭") from error
            try:
                return pending.result(timeout=self.timeout + 2)
            except (FutureTimeout, TimeoutError) as error:
                pending.cancel()
                raise AgentError(f"MCP {self.prefix} 请求超时，未自动重试；远端操作可能已执行") from error
            except Exception as error:
                raise AgentError(f"MCP {self.prefix} 请求失败：{error}") from error
            except BaseException:
                pending.cancel()
                raise

    def list_tools(self) -> list[dict]:
        async def discover(session):
            tools, cursor, seen = [], None, set()
            while True:
                page = await session.list_tools(cursor=cursor)
                tools.extend(tool.model_dump(mode="json", by_alias=True, exclude_none=True)
                             for tool in page.tools)
                cursor = page.nextCursor
                if not cursor:
                    return tools
                if cursor in seen:
                    raise ValueError("MCP 工具列表返回重复分页游标")
                seen.add(cursor)
        return self._request(discover)

    def call_tool(self, name: str, arguments: dict) -> dict:
        result = self._request(lambda session: session.call_tool(name, arguments))
        data = result.model_dump(mode="json", by_alias=True, exclude_none=True)
        if result.isError:
            detail = "\n".join(item.text for item in result.content if item.type == "text")
            raise AgentError(f"MCP 工具 {name} 执行失败：{detail or '服务端返回 isError'}")
        return data

    def register_tools(self, registry, *, allowlist=None) -> list[str]:
        """发现后逐个注册；allowlist 使用服务端原名。注册前检查全部名称。"""
        if allowlist is not None and (not isinstance(allowlist, list) or
                                      any(not isinstance(name, str) for name in allowlist)):
            raise ValueError("MCP tools 必须是工具原名的字符串列表")
        discovered = self.list_tools()
        available = {tool["name"] for tool in discovered}
        if allowlist is not None and set(allowlist) - available:
            raise ValueError("MCP 未找到指定工具：" + ", ".join(sorted(set(allowlist) - available)))
        selected = [tool for tool in discovered if allowlist is None or tool["name"] in allowlist]
        adapted = [_MCPTool(self, tool) for tool in selected]
        names = [tool.name for tool in adapted]
        existing = {item["function"]["name"] for item in registry.schemas()}
        if len(names) != len(set(names)) or existing.intersection(names):
            raise ValueError("MCP 工具名称重复，请使用不同的服务前缀；本批次未注册")
        for tool in adapted:
            registry.register(tool)
        return names

    def close(self):
        with self._lock:
            self._closed = True
            if self._thread is None:
                return
            if self._loop is not None and self._thread.is_alive():
                try:
                    callback = self._stop.set if self._ready.done() else self._owner.cancel
                    self._loop.call_soon_threadsafe(callback)
                except RuntimeError:
                    pass  # 服务已退出，事件循环可能先于 close 结束。
            self._thread.join(timeout=self.timeout + 5)
            if self._thread.is_alive():
                raise AgentError(f"MCP {self.prefix} 连接关闭超时")

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()


class _MCPTool(Tool):
    """远端 Schema 无需反向生成 Python 函数或 Pydantic 模型。"""

    def __init__(self, client, definition):
        self.client = client
        self.remote_name = definition["name"]
        name = f"{client.prefix}__{self.remote_name}"
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            slug = re.sub(r"[^A-Za-z0-9_-]", "_", name)
            name = slug[:55] + "_" + sha256(name.encode()).hexdigest()[:8]
        self.name = name
        self.description = definition.get("description") or self.remote_name
        self.parameters = deepcopy(definition["inputSchema"])
        _, _, _, validator_for, Registry = client._deps
        try:
            validator = validator_for(self.parameters)
            validator.check_schema(self.parameters)
        except Exception as error:
            raise ValueError(f"MCP 工具 {self.remote_name} 的 inputSchema 无效：{error}") from error
        # 只解析 Schema 内部引用，不自动下载外部 $ref。
        self._validator = validator(self.parameters, registry=Registry())

    def schema(self):
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": deepcopy(self.parameters),
        }}

    def run(self, arguments):
        json.dumps(arguments, ensure_ascii=False, allow_nan=False)
        self._validator.validate(arguments)
        return self.execute(**arguments)

    def execute(self, **arguments):
        return self.client.call_tool(self.remote_name, arguments)


def load_mcp_config(path) -> dict:
    """读取 {mcpServers: {服务名: 配置}}；只支持明确配置的 stdio 命令。"""
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {"mcpServers"}:
        raise ValueError("MCP 配置顶层需要且只接受 mcpServers")
    servers = config["mcpServers"]
    if not isinstance(servers, dict):
        raise ValueError("mcpServers 必须是服务名到配置的映射")
    allowed = {"command", "args", "cwd", "env", "timeout", "tools"}
    for name, options in servers.items():
        if not isinstance(options, dict) or set(options) - allowed or "command" not in options:
            raise ValueError(f"MCP 服务 {name} 配置无效；仅支持 command/args/cwd/env/timeout/tools")
        if "cwd" in options:
            if not isinstance(options["cwd"], str):
                raise ValueError("MCP cwd 必须是目录路径")
            options["cwd"] = str((path.parent / Path(options["cwd"]).expanduser()).resolve())
    return servers
