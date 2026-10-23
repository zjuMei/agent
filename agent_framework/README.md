# 从零实现自己的 Agent 框架

参考 [Hello-Agents 第七章：构建你的 Agent 框架](https://hello-agents.datawhale.cc/#/./chapter7/第七章%20构建你的Agent框架)。
这是本仓库独立实现的教学框架，使用 OpenAI SDK 调用兼容模型，不依赖 `hello-agents`、LangChain 或 LangGraph。

与前两版旅行助手相比，这里的核心变化是：**流程不再绑定旅行业务，业务能力通过注册工具加入。**
默认示例使用计算器，方便专注理解框架；可以注册自己的天气、搜索或其他工具。
当前还提供第八章的轻量扩展：SQLite 会话持久化、显式长期记忆和本地文档检索。
基础扩展只使用 Python 标准库，不需要额外数据库服务或 Embedding 密钥。

## 先运行

Python 3.10+。以下命令都在仓库根目录 `hello_agent/` 执行：

```bash
python -m pip install -r agent_framework/requirements.txt

# 无需 API Key：预设模型回复演示控制流，计算器实际执行
python -m agent_framework.examples.offline_demo

# 使用根目录现有 .env 的真实模型
python -m agent_framework --mode react --trace "请调用计算器计算 sqrt(16) + 2 * 3"

# 连续对话，默认 react、流式输出和会话持久化
python -m agent_framework

# 关闭流式输出，等待完整流程完成后一次性显示答案
python -m agent_framework --no-stream "用两句话解释什么是 Agent"

# 临时会话：本次进程退出后不保留数据
python -m agent_framework --db :memory:
```

已有 `.env` 直接复用；首次使用时从根目录 `.env.example` 复制并填写，避免覆盖已有配置。
运行中环境变量优先于 `.env`，框架内部不会自动读取配置文件。

| 环境变量 | 用途 |
| --- | --- |
| `LLM_API_KEY` | 模型服务密钥，也可使用 `OPENAI_API_KEY` |
| `LLM_MODEL` | 模型名称，兼容 `LLM_MODEL_ID` |
| `LLM_BASE_URL` | 兼容 Chat Completions 的服务地址；留空使用 SDK 默认地址 |
| `LLM_TIMEOUT` | 每个请求超时，默认 60 秒；SDK 最多重试 1 次 |
| `LLM_TEMPERATURE` | 默认 0.2 |
| `LLM_MAX_COMPLETION_TOKENS` | 默认 1536 |
| `LLM_ENABLE_THINKING` | 可选服务商扩展参数；配置后才发送，不兼容时删除此项 |

本地 Ollama/vLLM 也通过 `LLM_BASE_URL` 接入，填写服务中实际部署的模型名和非空占位密钥。
本框架使用显式地址配置，不按密钥格式推断服务商。模型必须支持上表中的生成参数；使用工具还需支持 Function Calling。

## 四种流程

| 模式 | 类 | 执行顺序 | 工具 |
| --- | --- | --- | --- |
| `simple` | `SimpleAgent` | 对话 → 回答 | 不使用 |
| `react` | `ReActAgent` | 模型 → 工具 → 模型，循环直到回答 | 使用注册工具 |
| `reflection` | `ReflectionAgent` | 草稿 → 评审 → 按需修订 | 不使用 |
| `plan` | `PlanAndSolveAgent` | 规划 → 逐步执行 → 汇总 | 执行阶段使用 |

```bash
python -m agent_framework --mode simple "用两句话解释什么是 Agent"
python -m agent_framework --mode reflection --trace "给初学者写一段 Python 函数的说明"
python -m agent_framework --mode plan --trace "周一卖15个苹果，周二两倍，周三比周二少5个，计算三天总数"
```

默认开启模型流式请求，终端按 `[回答]`、`[草稿]`、`[评审]`、`[规划]`、`[步骤 N]`、`[汇总]` 等阶段实时显示文本。
规划和评审的 JSON 也会显示，方便观察完整流程；Reflection 最后会明确给出最终答案。
工具名在模型调用完整接收后显示，工具结果在实际执行后显示。
`--trace` 在本轮结束后额外显示完整模型回复、调用参数、工具结果、错误及耗时。
这里展示的是可观察的调用记录，不要求模型输出内部思维过程。

## 流式接口与完整返回值

`agent.run()` 默认使用流式模型请求，但仍返回完整答案字符串。
模型接收与终端显示分开：应用通过 `on_event` 回调获取实时片段；不传回调时不会自动打印。
CLI 已接入实时显示，因此直接运行 `python -m agent_framework` 即为流式输出。

```python
def show(event):
    if event["type"] == "model_start":
        print(f"\n[{event['stage']}] ", end="", flush=True)
    elif event["type"] == "text_delta":
        print(event["text"], end="", flush=True)
    elif event["type"] == "model_end":
        print()

answer = agent.run("用两句话解释 Agent", on_event=show)
# answer 是完整答案，历史记录也保存完整消息。

# 需要完整响应模式时显式关闭流式请求：
answer = agent.run("再给一个例子", stream=False)
```

事件包括 `model_start`、`text_delta`、`model_end` 和 `tool_result`，均携带阶段名称 `stage`。
`text_delta` 携带 `text`；`model_end` 携带完整 `message`；`tool_result` 携带工具名称、成功状态及结果或错误。
直接使用 `LLM.chat()` 时同样默认流式，可通过 `on_token(text)` 接收文本片段，返回值仍为完整 `Message`。
自定义 `ChatModel` 需要接受 `tools`、`stream`、`on_token` 这三个关键字参数，示例见 `ScriptedLLM`。
流式工具参数按调用序号拼接，等正常结束并完成校验后才执行工具。
中断、截断或格式错误时不执行不完整调用，也不把失败轮次保存到会话；屏幕上已显示的片段不会撤回。
分片拼接方式参考 [OpenAI Docs：Function calling](https://developers.openai.com/api/docs/guides/function-calling#streaming)。

## 第八章的基础扩展

CLI 默认把成功会话保存到 `agent_framework/data/agent.db`，重启后恢复相同用户、会话和模式的记录。
通过 `--user`、`--session` 分组，`--db :memory:` 可使用本次进程的临时数据。
长期记忆使用 `/remember 内容` 明确写入，`/memories [关键词]` 查看，`/update 记忆ID 内容` 更新，`/forget 记忆ID` 删除。
`/clear` 仅清除当前会话和模式的聊天记录。

```bash
# 无需密钥：演示真实 SQLite、检索及数据库重开后的恢复
python -m agent_framework.examples.knowledge_demo

# 导入随附 Markdown 示例后进入连续对话，默认流式输出
python -m agent_framework --ingest agent_framework/knowledge --session learning

# 仅导入，不调用模型
python -m agent_framework --ingest agent_framework/knowledge --ingest-only
```

CLI 的 ReAct/Plan 默认增加只读工具 `search_memory` 和 `search_documents`。
文档检索支持 UTF-8 的 Markdown/TXT，返回片段、行号及来源；采用英文词和中文双字匹配，无需 Embedding。
完整接口、隔离规则、示例及阅读顺序见 [基础扩展说明](EXTENSIONS.md)。

## 目录与阅读顺序

```text
agent_framework/
├── __main__.py               # python -m agent_framework 的入口
├── demo.py                   # 配置加载、交互命令、模式选择
├── requirements.txt
├── mini_agents/
│   ├── core/
│   │   ├── message.py        # Message / ToolCall，与 API 格式互转
│   │   ├── config.py         # 环境变量与参数校验
│   │   ├── llm.py            # ChatModel 接口 / OpenAI SDK 适配器
│   │   ├── agent.py          # 统一 run、会话与执行记录
│   │   ├── storage.py        # SQLite 连接、表结构和事务
│   │   ├── retrieval.py      # 基础文本匹配与长度限制
│   │   └── exceptions.py     # 模型、输出和步数异常
│   ├── memory/              # SessionStore / MemoryStore
│   ├── retrieval/           # 文档切分、导入和检索
│   ├── agents/
│   │   ├── simple.py
│   │   ├── react.py
│   │   ├── reflection.py
│   │   └── plan_solve.py
│   └── tools/
│       ├── base.py           # Tool 基类、FunctionTool 函数适配器
│       ├── registry.py       # 注册、Schema、统一执行入口
│       ├── builtin.py        # AST 计算器、可选 Tavily 搜索
│       ├── knowledge.py      # 长期记忆与文档检索工具
│       └── chain.py          # ToolChain / AsyncToolExecutor
├── examples/offline_demo.py  # 可替换的预设模型，展示四种流程
├── examples/knowledge_demo.py # 离线验证记忆、检索和恢复
├── knowledge/framework.md    # 可导入的示例资料
├── data/                    # 运行时数据库，Git 忽略
└── tests/                   # 离线测试，包括 SDK 的模拟 HTTP 和 SSE 测试
```

建议先看 `message.py → llm.py → agent.py → simple.py`，理解最小对话框架。
再看 `tools/base.py → tools/registry.py → agents/react.py`，理解工具闭环。
最后看 Reflection、Plan-and-Solve 和工具组合，比较不同类如何复用公共组件。

## 用框架写自己的应用

在仓库根目录的新 Python 文件中，可以这样使用：

```python
from pathlib import Path
from dotenv import load_dotenv
from agent_framework.mini_agents import Config, LLM, ReActAgent, create_registry

load_dotenv(Path(__file__).resolve().parent / ".env")
tools = create_registry()

@tools.tool
def get_course(name: str) -> dict:
    """查询示例课程；name 为课程名称。"""
    return {"name": name, "duration_minutes": 45}

llm = LLM(Config.from_env())
try:
    agent = ReActAgent(llm, tools=tools, system_prompt="你是课程助手，查询课程时使用工具。")
    print(agent.run("查询 Python 课程的时长，再算一下连续学习 3 节课需要多久"))
    print(agent.run("那两节呢？"))
finally:
    llm.close()
```

函数参数必须带类型注解，返回值应能序列化为 JSON。
注册器根据函数签名生成 JSON Schema，保留默认值，并拒绝缺少参数、多余参数及不匹配的类型。
需要维护客户端或状态时，继承 `Tool`，提供 `name`、`description`、Pydantic `input_model`，实现 `execute()`，然后 `tools.register(instance)`。
新增 Agent 类型则继承 `Agent`，实现 `_run(messages)`，将最终回答追加到消息列表并返回字符串。
调用模型时复用 `_chat()` / `_text()`，即可沿用流式事件与执行记录，无需为四种流程分别实现流式版本。

要使用网络搜索，额外安装依赖并配置 `TAVILY_API_KEY`：

```bash
python -m pip install tavily-python
python -m agent_framework --mode react --search --trace "搜索 Python 官方最新发布信息，附来源"
```

只有指定 `--search` 才注册 Tavily 联网搜索工具。搜索无结果或失败时直接返回真实状态，不生成模拟资料。

## 工具组合

`ToolChain` 适合事先确定的顺序流程；每一步输出写入上下文，下一步顶层字符串参数通过 `{变量名}` 引用它：

```python
from agent_framework.mini_agents import ChainStep, ToolChain, create_registry

chain = ToolChain(create_registry(), [
    ChainStep("calculator", {"expression": "{input} * 2"}, "doubled"),
    ChainStep("calculator", {"expression": "{doubled} + 1"}, "answer"),
])
print(chain.run(input=3)["answer"])  # 7
```

`AsyncToolExecutor.run([(工具名, 参数字典), ...])` 在线程中并发执行互不依赖的同步工具，按输入顺序返回结果。
有依赖或会竞争同一资源的工具使用顺序流程；框架不会自动判断依赖。完整示例见离线演示。

## 运行边界

- ReAct 使用原生 Function Calling，一次只执行一个调用；工具结果携带原调用 ID。
- 默认每次工具循环最多调用模型 6 次，包含最终回答；达到上限抛出 `StepLimitError`，已执行工具的副作用不会撤销。
- Plan 默认最多 5 个步骤，每步独立受 6 次调用限制，总上限为 `1 + 5 × 6 + 1 = 32` 次。计划必须是合法 JSON 字符串数组，规划和汇总不执行工具。
- Reflection 默认最多评审 2 轮，最多调用模型 5 次。通过评审可提前停止；修订达到上限时返回最后版本，最终版本可能尚未再次评审。评审通过也不保证事实正确。
- 每个 Agent 的上下文默认保留最近 5 个成功轮次。CLI 自动持久化；Python API 在传入 `SessionStore` 后持久化，相同用户/会话/Agent 类型可恢复。ReAct 保存完整工具消息；Plan/Reflection 保存原始问答，内部步骤通过本轮 `trace` 查看。同一实例用于顺序调用。
- 模型请求失败或回复被截断会报错，失败轮次不写入会话。工具参数或执行失败转成 `ok: false` 的工具结果，交由模型读取。
- 计算器只解释白名单 AST 节点，并限制表达式复杂度和数值范围，不使用 `eval`。

这是第七章框架加第八章的基础扩展，提供统一模型接口、四种流程、工具系统、持久记忆和基础 RAG。
Embedding 语义检索、自动摘要/记忆提取、知识图谱、多模态和多 Agent 协作暂未加入。

## 验证

```bash
python -m unittest discover -s agent_framework/tests -t . -v
python -m agent_framework.examples.offline_demo
python -m agent_framework.examples.knowledge_demo
```

测试不联网、不读取真实密钥，验证工具校验与失败、完整消息协议、会话截断、步数限制、
计划解析、反思分支、工具链、并发结果，以及 OpenAI SDK 请求和响应的转换。
流式测试还验证文本在完成前回传、工具 ID/名称/参数分片拼接、正常结束前不执行工具、
连接中断与用户中断时释放连接、失败轮次不保存，以及非流式兼容。
扩展测试使用临时数据库，验证重开恢复、用户/会话/模式隔离、记忆维护、上下文长度、
文档切分与引用、重复导入替换、只读工具和 RAG 调用闭环。
离线演示用于验证代码流程；真实模型的任务完成质量需结合具体模型和问题判断。
