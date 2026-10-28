# 第九章：上下文预算、任务笔记与分段阅读

第一轮增加轻量 `ContextBuilder`、输入与工具结果预算，以及每次模型调用的上下文统计。
第二轮增加简单的任务笔记和按片段 ID 分段阅读。默认流式输出继续保留，未加入自动摘要或通用终端工具。
实现参考 [Hello-Agents 第九章：上下文工程](https://github.com/datawhalechina/hello-agents/blob/main/docs/chapter9/第九章%20上下文工程.md)。

## 先运行

在仓库根目录执行：

```bash
# 无需密钥：查看旧历史被移除、长工具结果变成预览，而 SQLite 仍保留原文
python -m agent_framework.examples.context_demo

# 真实模型：输入总预算 6000 字符，每条工具结果最多发送 1000 字符
python -m agent_framework --db :memory: --trace \
  --context-chars 6000 --tool-result-chars 1000 \
  "请调用 calculator 计算 sqrt(16) + 2 * 3"

# 文档检索也经过相同的预算入口
python -m agent_framework --ingest agent_framework/knowledge --trace \
  --context-chars 6000 --tool-result-chars 1000 \
  "请检索本地文档，解释如何注册工具，并引用来源"
```

## 预算定义

| Python 配置 | CLI 参数 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `max_input_chars` | `--context-chars` | 24000 | 本次消息和工具 Schema 的规范化 JSON 总字符数 |
| `max_memory_chars` | `--memory-chars` | 2000 | 自动注入的记忆正文总字符数，0 表示不自动发送 |
| `max_tool_result_chars` | `--tool-result-chars` | 3000 | 每条 tool 消息正文的字符上限，最小 128 |

计数使用 `ensure_ascii=False`、紧凑分隔符的 JSON，包含角色、工具 ID、调用参数及工具 Schema。
它不计 HTTP 头、生成参数和服务端的提示模板，既不是字节数，也不是精确 token 数。
`LLM_MAX_COMPLETION_TOKENS` 仍然单独控制输出 token 上限，默认 1536。
选择输入预算时还要为模型输出留出实际上下文空间；字符预算不能保证适配任意模型的 token 窗口。

记忆候选继续沿用第八章的规则：最多三条相关记忆，加最多三条最近记忆、去重，候选正文最多 2000 字符。
新配置限制的是实际发送上限；调大 `max_memory_chars` 不会扩大原有的候选收集范围。

## 每次调用前做什么

```text
Agent.run() 收集原始历史、记忆候选和当前任务笔记
    ↓
Simple / ReAct / Reflection / Plan 执行自己的流程
    ↓
Agent._chat() → ContextBuilder.build()
    1. 检查工具请求和结果的对应关系
    2. 复制消息，为过长工具结果生成结构化预览
    3. 按记忆正文预算选择候选
    4. 若总输入仍超限，依次移除最早的完整历史轮次
    5. 若仍超限，移除低优先级的记忆（候选顺序靠后的先移除）
    6. 若仍超限，本次不发送任务笔记，数据库中的笔记保留
    7. 必要输入仍超限，抛出 ContextLimitError，不调用模型
    ↓
LLM.chat() 默认流式请求
```

历史始终保留时间顺序，选择结果是最近若干完整轮次。
系统提示、当前用户问题、当前流程中的规划/步骤/评审指令、草稿和工具调用参数都不截断。
当前执行中的工具结果也受每条工具正文预算限制，但请求 ID、结果 ID 和对应关系保留。
当前流程本身过长时，本轮明确失败；本版不会用摘要替代它。

选择每次都从原始数据的副本开始，前一次裁剪不会永久丢失后续可能需要的历史。
SQLite 中保存的成功轮次、`agent.get_history()`、工具执行记录和 `tool_result` 事件仍保留原始工具结果。
输入预算只约束发送给模型的内容；终端工具结果展示及完整 trace 可能更长。
上下文超限的轮次不保存；如果本轮已经执行了工具，其副作用仍不会回滚。

## 工具结果预览

短结果保持原样。长 JSON 结果按结构处理：缩短正文字符串、限制列表条数/字典字段和嵌套深度，再重新序列化。
不会从序列化后的 JSON 中间直接切断，因此发给模型的预览仍是合法 JSON。
预览包含 `truncated: true` 和 `original_chars`，列表结果还提供 `total_items`。
成功/失败的 `ok` 状态保留；`source`、`citation`、`chunk_id`、行号等引用字段保持原样。
如果连引用本身都无法放进预算，则省略正文并明确说明，完整结果仍在原始记录中。
文档片段可用第二轮新增的 `read_chunk` 继续读取；其他任意工具的原始结果还没有通用读取接口。

## 任务笔记：每个会话一份当前进度

```bash
# 无需模型密钥：明确更新当前会话的整份笔记
python -m agent_framework --session learning "/note 目标：学习 Agent；已完成：工具注册；下一步：RAG"
python -m agent_framework --session learning "/notes"

# 重启后自动将笔记作为参考上下文，需现有模型配置
python -m agent_framework --session learning --trace "当前任务下一步是什么？"
```

交互模式也支持 `/note 内容`、`/notes`、`/note-clear`。
`/note` 会替换整份当前笔记，最多 2000 字符；可以写目标、已完成、问题、下一步和依据。
Python 中使用 `NoteStore.save()` 可以直接保存多行 Markdown，`get()` 读取，`clear()` 清除。
模型不能写笔记，不会自动从聊天中提取或更新进度。

笔记按 `user_id + session_id` 隔离，同一会话的不同 Agent 模式共享笔记；不同会话不共享。
CLI 的四种 Agent 都自动加载，Python API 需要传入 `notes=NoteStore(...)`。
笔记被标成历史参考数据，不写入用户长期偏好，也不重复保存到聊天轮次中。
`/clear` 清除聊天记录并保留笔记；清笔记用 `/note-clear`，不影响长期记忆、文档和其他会话。
`/note-clear` 不会删除旧聊天中已经引用过的进度；需要清除聊天内容时另用 `/clear`。
预算不足时优先移除旧历史、再移除记忆候选、最后尝试移除任务笔记。
统计中的 `note_present` 表示存在笔记，`note_included` 表示本次实际发送了笔记，`note_chars` 是已发送正文长度。

## 分段阅读：搜索预览，再读取正文

ReAct/Plan 的 `search_documents(query)` 现在对每个命中返回最多 200 字符预览。
结果带 `chunk_id`、`offset`、`next_offset`、`total_chars` 和来源；`DocumentIndex.search()` 本身仍保留原来的完整检索接口。

```text
search_documents(query="课程安排")
    → 得到片段 ID、前 200 字符和 next_offset
read_chunk(chunk_id="实际返回的片段ID", offset=200, max_chars=1000)
    → 读取指定部分；按这次实际返回的 next_offset 继续
    → next_offset 为 null 时到达片段末尾
```

`offset` 相对当前片段正文从零开始，单位为 Python 字符；`max_chars` 默认 1000，范围为 1 至 2000。
`citation` 和行号仍描述原始片段的范围，页内位置通过 `offset` 表示。
读取的对象是 SQLite 中已经索引的片段，不提供任意路径读取。用户/命名空间由应用绑定，模型不能切换。
重新导入文档后旧片段 ID 可能失效，此时工具返回真实错误，应重新搜索。

每页仍经过统一输入及工具结果预算。若正文被预算进一步缩短，返回的 `next_offset` 同步改为实际可见正文的结尾，
正文保留精确原文切片，避免跳过未读字符。如果预算小到连必要元数据都容纳不了，会返回省略提示，需要调大工具结果预算。
连续读取仍受 ReAct 的步数上限和当前执行消息总预算限制，应按需读取。

```bash
# 离线：真实搜索、分段读取、笔记保存及数据库重开恢复；模型回复仍是预设数据
python -m agent_framework.examples.knowledge_demo
```

## 查看统计

使用 `--trace`，每条模型调用记录会增加 `context`，失败记录也会在可用时带上统计：

```json
{
  "unit": "characters",
  "budget_chars": 6000,
  "input_chars_before": 9000,
  "input_chars": 3500,
  "tool_schema_chars": 800,
  "history_turns_total": 5,
  "history_turns_kept": 3,
  "history_turns_dropped": 2,
  "memory_ids_kept": ["memory_id"],
  "memory_ids_dropped": [],
  "memory_ids_truncated": [],
  "note_present": true,
  "note_included": true,
  "note_chars": 60,
  "tool_results_trimmed": [
    {"message_index": 5, "tool_call_id": "call_id", "original_chars": 5000, "sent_chars": 900}
  ],
  "reasons": ["输入超预算，移除较早的完整历史轮次", "过长工具结果改为结构化预览，原始记录保留"]
}
```

上面是字段示意，数值取决于实际输入。`message_index` 是裁剪前消息列表的零起始位置。
`input_chars_before` 是原始消息加任务笔记及全部收集到的记忆候选的大小；`input_chars` 是最终实际发送大小。
`tool_results_trimmed` 仅列出最终仍被发送的工具结果；已经随旧轮次移除的结果不会列入。
`stage=记忆` 的记录展示收集到的候选，实际发送哪些由每次调用的 `context` 字段说明。
`model_start` 流式事件也带有同一份统计，原有事件类型保持不变。

## Python 接口

```python
from agent_framework.mini_agents import ContextBuilder, ContextConfig, NoteStore, ReActAgent

# llm、tools、db 沿用框架的模型、注册表和 SQLiteStore；默认仍流式请求。
agent = ReActAgent(
    llm, tools=tools,
    notes=NoteStore(db, user_id="local", session_id="learning"),
    context=ContextBuilder(ContextConfig(
        max_input_chars=6000,
        max_memory_chars=1000,
        max_tool_result_chars=1000,
    )),
)
answer = agent.run("计算 2 + 3")
print(agent.trace[-1]["context"])
```

新增 Agent 继续使用 `_chat()` 或 `_text()`，即可复用预算和统计。
传入的消息需要保持 `system → 原始历史完整轮次 → 当前任务/内部执行消息` 的前缀结构；不符合时会明确报错。
直接调用底层 `LLM.chat()` 不经过 Agent 的 ContextBuilder。

## 验证与阅读顺序

```bash
python -m unittest discover -s agent_framework/tests -t . -v
python -m agent_framework.examples.context_demo
```

先看 `core/context.py` 的 `ContextConfig` 和 `ContextBuilder.build()`，再看 `core/agent.py` 中的 `_chat()`。
测试覆盖完整轮次选择、引用保留、合法 JSON、输入总预算、每次内部调用重新检查、原始数据保留及超限不调用模型。
