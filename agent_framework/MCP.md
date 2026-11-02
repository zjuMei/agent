# 第十章：调用 MCP Server 的工具

这次只增加 MCP 客户端扩展。ReAct/Plan 可以同时使用本地工具和 MCP Server 提供的工具。
先支持本地 `stdio`：框架启动配置中的程序，通过标准输入/输出通信，退出时关闭连接和子进程。

## 安装与快速验证

以下命令在仓库根目录运行，使用同一个 Python 环境：

```bash
python -m pip install -r agent_framework/requirements-mcp.txt

# 无需模型密钥：预设模型回复，实际启动 MCP 子进程、发现工具并执行
python -m agent_framework.examples.mcp_demo

# 无需模型密钥：查看本地工具和 MCP 工具的完整 Schema
python -m agent_framework --db :memory: \
  --mcp-config agent_framework/examples/mcp_servers.json --list-tools

# 使用根目录 .env 中的真实模型，默认流式输出
python -m agent_framework --db :memory: --trace \
  --mcp-config agent_framework/examples/mcp_servers.json \
  "请调用课程工具，查询 Agent 入门课程的时长"

# 同样可以交给 PlanAndSolveAgent
python -m agent_framework --mode plan --db :memory: \
  --mcp-config agent_framework/examples/mcp_servers.json \
  "查询 Agent 入门课程时长，再调用加法工具计算两节课的总分钟数"
```

示例课程工具实际返回 `duration_minutes: 45`。模型调用名是 `courses__get_course`，
加法工具是 `courses__add`；它们和计算器一样经过注册表、执行记录与上下文预算。
使用 `--no-stream` 可以关闭流式显示。

基础 `requirements.txt` 不增加 MCP 依赖。未安装 MCP 仍可正常使用原有功能；
尝试连接时才提示安装可选依赖。
为保留本框架的 Python 3.10 支持，这里使用官方 SDK 的 1.x 接口并限定 `mcp>=1.28,<2`。
SDK 2.x 的接口不同，不适用于本扩展。

## 连接自己的服务

参考 `examples/mcp_servers.json`：

```json
{
  "mcpServers": {
    "courses": {
      "command": "python",
      "args": ["-m", "agent_framework.examples.mcp_server"],
      "cwd": "../..",
      "timeout": 30,
      "tools": ["get_course", "add"]
    }
  }
}
```

| 字段 | 含义 |
| --- | --- |
| `courses` | 服务前缀，1～24 个字母、数字、下划线或连字符 |
| `command` | 可执行程序名或绝对路径，例如虚拟环境中的 Python |
| `args` | 参数列表，逐项传递，不作为 Shell 命令执行 |
| `cwd` | 可选工作目录；相对路径以配置文件目录为基准，省略时继承当前目录 |
| `env` | 可选的额外环境变量，值为字符串；在 SDK 的基础环境上增加 |
| `timeout` | 初始化、工具发现及每次调用的超时秒数，默认 30 |
| `tools` | 可选的工具原名列表；省略时全部接入，空列表不接入任何工具 |

可以在 `mcpServers` 下配置多个服务，它们使用独立连接和前缀。
`command` 不会自动切换 Python 环境；找不到 `mcp` 时，请确认服务使用的是安装了依赖的解释器。
`env` 中的值按字面传递，不展开 `${变量}`；模型配置 `.env` 不会自动传给服务进程。
服务的 stdout 用于协议消息，普通日志应写到 stderr。

`tools` 限定向模型提供的能力，不改变服务自身的访问权限。
连接或发现失败时明确退出，不会悄悄省略失败的服务。
`--list-tools` 可以先验证配置和工具 Schema，再配置或调用模型。

## Python 接口

不调用模型，也可以独立验证服务：

```python
import sys
from agent_framework.mini_agents import MCPClient, create_registry

tools = create_registry()
with MCPClient(
    sys.executable,
    ["-m", "agent_framework.examples.mcp_server"],
    prefix="courses",
) as client:
    names = client.register_tools(tools, allowlist=["get_course"])
    print(names)  # ["courses__get_course"]
    result = tools.execute("courses__get_course", {"name": "Agent 入门"})
    print(result)
    # 也可以把 tools 传给已有的 ReActAgent(llm, tools=tools)
    # 或 PlanAndSolveAgent(llm, tools=tools)，在 with 范围内完成 agent.run(...)。
```

`client.list_tools()` 返回服务端工具定义；`client.call_tool("get_course", {...})`
可以按原名直接调用。通常使用 `tools.execute()`，这样执行前还会依据服务端 JSON Schema
在本地校验参数。校验保留嵌套结构、必填字段及数值限制，不把字符串自动转为整数。

普通工具名加上前缀后直接使用，例如 `courses__get_course`。
含特殊字符或总长超过 64 的名称会转换为合法名称并附加短哈希，原名仍用于服务端调用。
`register_tools()` 的返回值或 `--list-tools` 展示最终名称。
名称冲突、白名单工具不存在或 Schema 无效时，本批次不注册；已有工具保留。

成功结果沿用注册表格式：

```text
{"ok": true, "result": {
  "content": [{"type": "text", "text": "服务端返回的内容"}],
  "isError": false
}}
```

若服务端提供 `structuredContent`，也会保留。服务返回 `isError: true`、连接异常、
超时或参数错误，会变为 `{"ok": false, "error": "..."}`，供模型判断下一步。
工具调用不自动重试；超时不能证明远端没有执行，因此不自动重复有副作用的操作。

## 如何接入现有框架

```text
MCPClient 连接并初始化
    → list_tools 获取名称、描述、inputSchema
    → 每个远程工具分别包装成 _MCPTool
    → ToolRegistry.schemas 交给模型
    → ReAct/Plan 选择工具
    → 注册表校验参数 → MCP call_tool
    → 工具结果进入 trace、流式事件及下一次模型调用
```

`_MCPTool` 覆盖 `schema()` 和 `run()`，直接使用远程 JSON Schema，
无需构造带动态签名的 Python 函数。它仍遵守现有 `Tool` 接口。
同步注册表通过一个后台事件循环访问异步 SDK；连接在整个 `with` 期间复用。
进入与退出 SDK 上下文的操作留在同一个异步任务里。调用串行提交，退出时等待 SDK 清理子进程。

工具结果仅在发送给模型的副本中受现有字符预算裁剪，完整执行记录继续保留。
所有 MCP 结果仍作为参考数据处理，不替换系统提示。过多的工具 Schema 也会消耗输入预算，
可以用 `tools` 只接入需要的能力。

## 范围与验证

当前支持 stdio 工具发现（含分页）和调用；不支持远程 HTTP/SSE、Resources、Prompts、
工具列表变更通知或自动重连。非文本内容保留为 JSON，不转换为模型原生图片消息。
本次没有增加 A2A、ANP 或多 Agent 协作。

```bash
python -m unittest discover -s agent_framework/tests -t .
```

安装 MCP 依赖后，测试会真实启动子进程，验证发现、复杂参数校验、ReAct/Plan 闭环、
工具错误、超时、服务退出、初始化失败和进程清理；无需模型密钥或联网。
未安装依赖时，真实 MCP 集成测试会明确跳过，其余测试继续运行。

阅读顺序：`examples/mcp_server.py` → `examples/mcp_demo.py` →
`mini_agents/tools/mcp.py` 中的 `_MCPTool`、`register_tools()` 和连接生命周期。

参考：[Hello-Agents 第十章](https://github.com/datawhalechina/hello-agents/blob/main/docs/chapter10/第十章%20智能体通信协议.md)、
[官方 MCP Python SDK 1.x](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x)。
