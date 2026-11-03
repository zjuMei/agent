# 本地资料研究助手

使用本仓库 `agent_framework/mini_agents` 的独立应用。输入研究问题和资料目录，
规划研究子问题、检索并阅读原文，输出带来源的 Markdown 报告。

复用现有模型配置，不需要搜索、地图或 Embedding 等额外 API，也没有新增依赖库。
资料在本地索引，读取的片段会发送给已配置的模型；模型是否在本地运行由现有配置决定。

## 运行

Python 3.10+，以下命令在仓库根目录执行：

```bash
python -m pip install -r local_research/requirements.txt

# 无需密钥或联网：预设模型回复，实际执行 SQLite、检索、阅读和报告检查
python -m local_research.demo

# 真实模型：复用根目录现有 .env，默认流式输出，使用随附教学资料
python -m local_research --max-output-tokens 2000 \
  "比较 ReAct、Reflection 和 Plan-and-Solve 的工作方式、适用场景与限制"

# 使用自己的 UTF-8 Markdown/TXT 资料
python -m local_research \
  --docs ./my_notes --output ./local_research/workspace \
  "根据这些资料，比较两种方案的优点、限制和适用条件"

# 关闭流式请求和过程显示，完成后显示正式报告
python -m local_research --no-stream "本地资料研究流程包含哪些步骤？"
```

已有框架依赖时无需重复安装。真实模型沿用 `LLM_MODEL`、`LLM_API_KEY`、`LLM_BASE_URL` 等配置，
首次配置方法见 [框架说明](../agent_framework/README.md)。进程环境变量优先于 `.env`，不会覆盖已有配置文件。

`knowledge/` 是随附教学资料，包含 ReAct、Reflection、Plan-and-Solve 与研究流程。
这些文件用于展示应用闭环，不是模型能力或性能的评测结论。
`demo` 明确使用预设回复，只验证控制流；正常入口使用真实模型。

## 使用流程

```text
问题 + 资料目录
    → 导入本次资料，建立独立索引
    → PlanAndSolveAgent 规划最多 3 个子问题
    → 逐步搜索 → 按需阅读原文 → 整理证据要点
    → 汇总 Markdown 草稿
    → 检查章节、引用编号和实际读取记录
    → 保存报告、证据、执行记录和任务状态
```

一个 `PlanAndSolveAgent` 完成整个研究，执行阶段复用框架的 ReAct 工具循环。
提示词要求计划围绕研究对象展开，汇总由最后阶段完成；模型仍可能产生多余步骤，可以从 trace 核对。
第一版不依赖 MCP 服务，不增加框架核心功能。

默认最多 3 个研究步骤，每步最多 6 次逻辑模型调用，加上规划和汇总，最多 20 次。
底层模型 SDK 的网络重试另计。可以用 `--max-plan-steps 1` 简化任务；
`--max-steps` 可设为 1～6，过小可能在得到答案前达到步数上限。

`--context-chars` 默认 24000，沿用框架字符预算；必要输入过长时明确失败，不无限增长。
`--max-output-tokens` 可覆盖本次模型输出上限；未指定时保留原有环境配置。
报告长度通过提示词引导，并不是程序强制的字数保证。

## 两个工具与证据编号

| 工具 | 行为 |
| --- | --- |
| `search_documents(query)` | 复用索引检索，返回最多 3 个命中及各自前 200 字符预览 |
| `read_chunk(chunk_id, offset=0, max_chars=1000)` | 读取搜索返回的片段，记录实际返回的非空原文 |

每个片段分配本次任务内稳定的 `S1`、`S2` 等编号，放在工具结果的 `citation` 字段里。
再次搜索相同片段会得到同一编号。来源使用相对资料目录的完整路径，因此不同目录的同名文件不会混淆。

搜索预览不算已读正文。报告引用某个编号前，必须实际调用 `read_chunk` 读取非空内容。
分段阅读使用返回的 `next_offset`；它为 null 时到达该片段末尾。
原始行号描述整个片段，各次实际读取的字符偏移和原文记录在 `evidence.json` 中。
只读过某一页不代表读完全部片段，检查也不声称覆盖所有原文。

引用必须写成独立的 `[S1]`，多个引用写成 `[S1][S2]`。
程序检查编号存在且对应正文被读取，并生成参考资料列表。它不验证每一句结论是否被原文支持，仍需人工核对。

## 输出文件与状态

每次研究创建一个独立任务目录，默认根目录为 `local_research/workspace/`：

```text
workspace/<任务ID>/
├── report.md       # 正式报告，或明确标注资料不足的结果
├── evidence.json   # 片段编号、来源、预览和实际读取的原文页面
├── trace.json      # 完整规划、模型回复、工具参数/结果和上下文统计
├── task.json       # 问题、资料范围、设置、计划、状态、时间和错误
├── index.sqlite3   # 本任务的索引快照，原文件变化不影响已导入内容
└── draft.md        # 检查失败时保留；没有读到证据时也保留模型草稿
```

| 状态 | 输出与退出码 |
| --- | --- |
| `completed` | 结构与引用检查通过，生成 `report.md`；退出码 0 |
| `insufficient` | 没有成功读取可引用的原文，生成明确说明资料不足的结果；退出码 2 |
| `validation_failed` | 有读取记录，但草稿结构或引用不合格，只保存 `draft.md`；退出码 1 |
| `failed` | 达到步数上限、模型/输入错误等，保留已取得的证据和执行记录；退出码 1 |
| `interrupted` | 用户中断，保留记录；退出码 130 |

空目录、没有支持的文件或只有空文件时不调用模型。
没有证据时，模型草稿不会作为有依据的研究报告发布。
执行失败不生成完成报告；引用失败也不会通过静默删除错误引用来伪装成功。

流式显示的是执行过程与草稿，最终以任务状态和保存的文件为准。
`--trace` 会额外在终端展示完整记录；不指定也始终保存 `trace.json`。
每个任务独立，没有跨任务聊天记忆，也不自动向个人长期记忆写入研究结论。

## Python 接口

```python
from dotenv import load_dotenv
from agent_framework.demo import StreamPrinter
from agent_framework.mini_agents import Config, LLM
from local_research import run_research

load_dotenv(".env", override=False)
llm = LLM(Config.from_env())
try:
    result = run_research(
        "比较三种 Agent 流程",
        docs="local_research/knowledge",
        output="local_research/workspace",
        llm=llm,
        on_event=StreamPrinter(),
    )
    print(result.status, result.report_path)
finally:
    llm.close()
```

`run_research()` 返回 `ResearchResult`，包含任务 ID、目录、状态、报告路径和错误。
调用方管理模型连接；`stream=False` 可关闭流式模型请求，事件回调是可选的。
无效问题、目录和限制参数在创建任务前报错；研究开始后的执行错误会保存到任务记录并返回失败状态。

## 范围与验证

第一版支持 UTF-8 `.md`、`.txt`，不直接解析 PDF、Word 或图片。
只导入本次指定目录的资料，跳过指向目录外的符号链接；输出目录位于资料目录下时，会排除输出内容。
每次任务使用独立索引，因此删除的旧文件和历史报告不会自动混入新研究。

检索沿用英文词和中文相邻双字匹配，同义表达可能匹配不到；可以换短关键词或补充资料。
它适合小型资料库，不保证穷尽所有资料。两份冲突资料都被检索和阅读后，模型应分别说明双方依据。
第一版没有中途断点续跑、自动语义评审或自动修复引用，失败后查看记录并重新发起任务。

```bash
# 应用专项测试，无需密钥或联网
python -m unittest discover -s local_research/tests -t . -v

# 框架和应用的全部测试；安装 MCP 依赖后也包含 MCP 集成测试
python -m unittest discover
```

测试覆盖真实索引与分页读取、同名文件来源、冲突资料证据保存、引用检查、空资料、无匹配、
旧文件隔离、流式开关、模型失败、上下文超限、步数上限及中断记录。

代码阅读顺序：`tools.py` → `report.py` → `researcher.py` → `__main__.py`。
`demo.py` 展示可观察的控制流，研究执行直接复用框架现有实现。
