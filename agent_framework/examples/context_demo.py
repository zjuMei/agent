"""无需密钥：预设模型回复，真实执行上下文裁剪、工具预览和 SQLite 保存。"""

import json

from ..mini_agents import (
    ContextBuilder, ContextConfig, ContextLimitError, MemoryStore, Message, ReActAgent,
    SessionStore, SimpleAgent, SQLiteStore, ToolCall, ToolRegistry,
)
from .offline_demo import ScriptedLLM


def main():
    print("离线上下文演示：模型回复为预设数据，工具、预算选择及保存实际执行。")
    with SQLiteStore(":memory:") as db:
        session = SessionStore(db)
        session.append("ReActAgent", [Message("user", "旧历史" * 2000), Message("assistant", "已收到")])
        memory = MemoryStore(db)
        memory.add("我喜欢简洁的 Python 示例。")
        tools = ToolRegistry()

        @tools.tool
        def example_text() -> dict:
            """返回一段较长的示例文本。"""
            return {"content": "Python 工具示例正文。" * 1000, "source": "示例文本"}

        llm = ScriptedLLM([
            Message("assistant", tool_calls=[ToolCall("text1", "example_text", "{}")]),
            "已读取工具结果预览，完整正文仍保存在原始记录中。",
        ])
        agent = ReActAgent(llm, tools=tools, memory=memory, session=session,
                           context=ContextBuilder(ContextConfig(max_input_chars=2000,
                                                                max_tool_result_chars=500)))

        def show(event):
            if event["type"] == "text_delta":
                print(event["text"], end="", flush=True)

        agent.run("请读取示例文本。", on_event=show)
        for record in agent.trace:
            if "context" in record:
                print(f"\n\n[{record['stage']}] 上下文统计：")
                print(json.dumps(record["context"], ensure_ascii=False, indent=2))
        sent = llm.requests[-1][0][-1]["content"]
        original = next(message.content for message in agent.get_history() if message.role == "tool")
        print(f"\n工具结果：实际发送 {len(sent)} 字符，历史中保留 {len(original)} 字符。")
        print(f"数据库仍保存 {len(session.load('ReActAgent'))} 个完整轮次。")

    model = ScriptedLLM([])
    agent = SimpleAgent(model, context=ContextBuilder(ContextConfig(max_input_chars=1)))
    try:
        agent.run("这个任务的必要输入超过预算")
    except ContextLimitError as error:
        print(f"\n必要输入超限：{error}")
        print(f"实际模型调用：{len(model.requests)} 次。")


if __name__ == "__main__":
    main()
