# 第八章的基础扩展：会话、记忆和文档检索

会话记录用于继续聊天，长期记忆保存明确的重要信息，知识库保存文档原文。
三者共用一个 SQLite 文件，通过现有 Agent 基类和工具注册表接入，不增加外部数据库或模型服务。
参考 [Hello-Agents 第八章：记忆与检索](https://github.com/datawhalechina/hello-agents/blob/main/docs/chapter8/第八章%20记忆与检索.md)，保留基础闭环，简化记忆类型和存储后端。
以下命令均在仓库根目录执行。

## 先运行

```bash
# 无需密钥：真实存储和检索，预设模型回复演示工具调用，再重新打开数据库验证恢复
python -m agent_framework.examples.knowledge_demo

# 导入随附示例；只导入时不需要密钥
python -m agent_framework --ingest agent_framework/knowledge --ingest-only

# 需要已有 .env 中的模型配置；默认流式输出
python -m agent_framework --session learning --trace "请检索本地文档，解释工具是如何注册的，并引用来源"
```

CLI 默认使用 `agent_framework/data/agent.db`，该目录已被 Git 忽略。
`--db PATH` 可指定其他文件；自定义数据库路径需要自行加入忽略规则。
`--db :memory:` 使用本次进程的临时数据库。

## 会话持久化

```bash
# 使用相同用户、会话和模式，即可在重启后继续聊
python -m agent_framework --user local --session learning

# 新会话共享当前用户的长期记忆和知识库，聊天记录独立
python -m agent_framework --user local --session another
```

会话按 `user_id + session_id + Agent 类型` 隔离，因此切换模式不会串用内部消息。
所有成功轮次保存在数据库中，模型每次最多接收最近 5 个完整轮次；实际数量还受 [上下文预算](CONTEXT.md) 限制。
ReAct 的工具请求、结果和调用 ID 一起保存；Plan/Reflection 仍只保存原始问答。
失败轮次不落盘，已执行工具的副作用不能回滚。
交互中的 `/clear` 清除当前用户、会话和模式的全部聊天记录，保留长期记忆、任务笔记与知识库。

## 显式长期记忆

在交互模式输入：

```text
/remember 我正在学习 Agent，喜欢简洁的 Python 示例
/memories
/memories Python
/update 记忆ID 我已经掌握工具注册，接下来学习 RAG
/forget 记忆ID
```

`/remember` 返回完整记忆 ID，更新和删除使用该 ID。相同用户重复保存相同内容不会产生重复记录。
这些命令由程序直接执行，不调用模型。也可以单独运行，完全不需要模型密钥：

```bash
python -m agent_framework "/remember 我喜欢简洁示例"
python -m agent_framework "/memories"
```

四种 Agent 每轮都会加载最多 3 条相关记忆并补充最多 3 条最近记忆，去重后正文共不超过 2000 字符。
这些是候选信息，每次实际发送时还会根据统一上下文预算选择，选择结果在 `trace` 的 `context` 字段中显示。
这些内容被标为参考数据，临时加入上下文，不再写入聊天记录。
这不是自动偏好提取或自动学习：自然语言说“请记住”时，CLI 提示模型介绍命令，持久写入使用 `/remember` 或应用的 `MemoryStore.add()`。
模型只获得 `search_memory(query)` 只读工具，不获得修改、删除记忆的工具。
删除长期记忆不会删除旧聊天中已经出现过的相同信息；需要清除会话时另行使用 `/clear`。

## 文档 RAG

```bash
# 导入自己的 UTF-8 .md/.txt 文件或目录后进入聊天；可重复指定 --ingest
python -m agent_framework --ingest ./my_notes --session notes
```

文件导入由应用执行；模型只能使用 `search_documents(query)` 检索已经导入的资料。
默认按段落或换行边界优先切分，每片最多 700 字符，窗口重叠参数为 80 字符。
长段落会继续切分；结果带 `source`、`chunk_id`、原始行号及 `citation`，例如 `framework.md:L1-L7`。
`DocumentIndex.search()` 每次最多返回 3 个片段、正文共 3000 字符，截断的内容会标记 `truncated`。
模型工具 `search_documents` 再为每个命中提供最多 200 字符预览，必要时用 `read_chunk` 分段读取正文，见 [第九章说明](CONTEXT.md)。
CLI 提示模型引用这些来源、对证据不足如实说明；引用是否正确仍需结合实际回答检查。

基础检索按英文词和中文相邻双字进行词频余弦匹配，不调用模型，也不需要 Embedding 服务。
只返回有词面交集的条目，无匹配时返回空数组；同义改写和复杂语义匹配能力有限。
同一路径重新导入会原子替换旧片段，包括清除空文件原来的内容。
导入目录会忽略不支持的文件及指向目录外的文件链接。
目录导入不会自动清理已从磁盘删除的旧文件，应用可调用 `DocumentIndex.remove(path)`。

CLI 的 ReAct/Plan 默认获得 `calculator`、`search_memory`、`search_documents`、`read_chunk` 四个工具，`--search` 另加 Tavily 搜索。
Simple/Reflection 会使用自动加载的记忆，但不执行检索工具；应用可先检索文档，再把资料放入它们的输入。
会话、记忆和知识库按 `--user` 隔离；这是本地数据分组方式，不是多用户登录认证。

## 在应用中接入

普通 Agent 默认仍只保存进程内历史。需要持久化时显式传入 `session`，需要自动加载记忆时传入 `memory`。
以下示例可保存为仓库根目录的 Python 文件运行：

```python
from dotenv import load_dotenv
from agent_framework.mini_agents import (
    Config, LLM, SQLiteStore, SessionStore, MemoryStore, DocumentIndex,
    ReActAgent, create_registry, register_knowledge_tools,
)

def show(event):
    if event["type"] == "model_start":
        print(f"\n[{event['stage']}] ", end="", flush=True)
    elif event["type"] == "text_delta":
        print(event["text"], end="", flush=True)

load_dotenv(".env")
llm = LLM(Config.from_env())
try:
    with SQLiteStore("agent_framework/data/agent.db") as db:
        memory = MemoryStore(db, user_id="local")
        memory.add("我正在学习 Agent，喜欢简洁示例")
        documents = DocumentIndex(db, namespace="local")
        documents.add_file("agent_framework/knowledge/framework.md")
        tools = register_knowledge_tools(create_registry(), memory=memory, documents=documents)
        agent = ReActAgent(
            llm, tools=tools, memory=memory,
            session=SessionStore(db, user_id="local", session_id="learning"),
            system_prompt="你是学习助手。查询本地文档时使用工具，并引用结果的 citation。",
        )
        answer = agent.run("根据文档解释工具注册", on_event=show)
finally:
    llm.close()
```

`MemoryStore` 提供 `add/list/search/update/delete`，`DocumentIndex` 提供 `add_file/add_directory/search/remove`。
用户标识由应用创建对象时绑定，模型不能通过工具参数切换用户。
数据库应保持打开直到 Agent 和工具用完；同一个 Agent 实例仍按顺序调用。
本地数据可能包含个人对话和记忆，按需备份数据库文件。

## 代码阅读顺序

1. `core/storage.py`：SQLite 表结构和事务。
2. `memory/session.py`：完整消息的保存和恢复。
3. `memory/store.py`：明确写入的长期记忆。
4. `retrieval/documents.py`：文本切分与原始行号。
5. `core/retrieval.py`、`retrieval/index.py`：基础匹配、长度限制和文档索引。
6. `tools/knowledge.py`：用熟悉的装饰器注册两个只读工具。
7. `core/agent.py`：运行前恢复上下文，成功后保存轮次。

本版未加入 Embedding 语义检索、自动摘要/记忆提取、知识图谱、多模态或 MQE/HyDE。
后续可替换检索评分部分，继续复用文档切分、来源记录和工具接口。
