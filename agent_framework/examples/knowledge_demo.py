"""无需密钥：真实 SQLite 和文档检索，模型回复用预设数据演示。"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ..demo import StreamPrinter
from ..mini_agents import (
    DocumentIndex, MemoryStore, Message, NoteStore, ReActAgent, SessionStore, SQLiteStore, ToolCall,
    create_registry, register_knowledge_tools,
)
from .offline_demo import ScriptedLLM


def main():
    print("离线记忆与检索演示：存储、切分、检索实际执行；模型回复为预设数据。")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "agent.db"
        with SQLiteStore(path) as db:
            memory = MemoryStore(db)
            notes = NoteStore(db)
            notes.save("目标：理解工具注册\n下一步：搜索文档并读取原文。")
            item = memory.add("我正在学习 Agent，喜欢简洁的 Python 示例。")
            print("\n已明确保存记忆：", item["content"])
            documents = DocumentIndex(db)
            source = Path(__file__).resolve().parents[1] / "knowledge" / "framework.md"
            print("已导入文档：", documents.add_file(source, chunk_size=240, overlap=30))
            hits = documents.search("工具注册")
            print("真实检索结果：", json.dumps(hits, ensure_ascii=False, indent=2))
            llm = ScriptedLLM([
                Message("assistant", tool_calls=[ToolCall(
                    "doc1", "search_documents", json.dumps({"query": "工具注册"}),
                )]),
                Message("assistant", tool_calls=[ToolCall(
                    "read1", "read_chunk", json.dumps({"chunk_id": hits[0]["chunk_id"], "offset": 200}),
                )]),
                "使用 @tools.tool 注册函数，参数需要类型注解。"
                f"[{hits[0]['citation']}]",
            ])
            tools = register_knowledge_tools(create_registry(), memory=memory, documents=documents)
            agent = ReActAgent(llm, tools=tools, session=SessionStore(db), memory=memory, notes=notes)
            printer = StreamPrinter()
            printer.finish(agent.run("请查询文档，解释如何注册工具。", on_event=printer))
            notes.save("目标：理解工具注册\n已完成：搜索预览并读取原文\n下一步：比较其他 Agent 流程。")

        # 关闭连接后重新打开，模拟程序重启。这里只检查恢复的数据，不假装验证模型理解能力。
        with SQLiteStore(path) as db:
            agent = ReActAgent(ScriptedLLM([]), session=SessionStore(db))
            print("\n重启后恢复的消息数：", len(agent.get_history()))
            print("重启后保留的记忆：", MemoryStore(db).list()[0]["content"])
            print("重启后保留的任务笔记：\n", NoteStore(db).get()["content"])
            print("重启后仍可检索：", DocumentIndex(db).search("工具注册")[0]["citation"])


if __name__ == "__main__":
    main()
