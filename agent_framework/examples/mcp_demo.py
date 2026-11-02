"""无需模型密钥：预设模型选择工具，实际启动并调用 MCP 子进程。"""

import json
import sys

from .offline_demo import ScriptedLLM
from ..mini_agents import MCPClient, Message, ReActAgent, ToolCall, create_registry


def main():
    print("MCP 演示：模型回复为预设数据，工具发现与执行使用真实 stdio 服务。")
    tools = create_registry()
    with MCPClient(sys.executable, ["-m", "agent_framework.examples.mcp_server"],
                   prefix="courses") as client:
        print("已接入：", client.register_tools(tools))
        call = Message("assistant", tool_calls=[
            ToolCall("course1", "courses__get_course", json.dumps({"name": "Agent 入门"})),
        ])
        llm = ScriptedLLM([call, "Agent 入门这门示例课程的时长为 45 分钟。"])
        agent = ReActAgent(llm, tools=tools)
        print(agent.run("查询 Agent 入门课程时长"))
        print("实际工具记录：", next(item for item in agent.trace if item["stage"] == "工具"))


if __name__ == "__main__":
    main()
