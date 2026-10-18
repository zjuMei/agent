"""固定回复只用来演示控制流，不调用真实模型，不需要 .env。"""

import asyncio

from ..mini_agents import (
    AsyncToolExecutor, ChainStep, Message, PlanAndSolveAgent, ReActAgent, ReflectionAgent,
    SimpleAgent, ToolCall, ToolChain, create_registry,
)


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def chat(self, messages, *, tools=None):
        self.requests.append(([m.to_dict() for m in messages], tools))
        reply = next(self.replies)
        return Message("assistant", reply) if isinstance(reply, str) else reply


def calculate_call(expression="2 + 3 * 4"):
    import json
    return Message("assistant", tool_calls=[
        ToolCall("call_calc", "calculator", json.dumps({"expression": expression})),
    ])


def main():
    print("离线控制流演示：模型回复为预设数据，计算器实际执行。")
    registry = create_registry()
    demos = [
        (SimpleAgent(ScriptedLLM(["你好，我是学习助手。"])), "你好"),
        (ReActAgent(ScriptedLLM([calculate_call(), "计算结果为 14。"]), tools=registry), "计算 2 + 3 * 4"),
        (ReflectionAgent(ScriptedLLM([
            "Agent 会调用模型。", '{"approved": false, "feedback": "补充工具调用和循环"}',
            "Agent 根据任务调用模型，按需执行工具，再根据工具结果决定继续或回答。",
            '{"approved": true, "feedback": ""}',
        ])), "用一句话解释 Agent"),
        (PlanAndSolveAgent(ScriptedLLM([
            '["计算表达式", "解释结果"]', calculate_call(), "结果为 14。",
            "先乘后加：3 * 4 = 12，2 + 12 = 14。", "答案是 14，先乘后加。",
        ]), tools=registry), "计算 2 + 3 * 4 并解释顺序"),
    ]
    for agent, question in demos:
        print(f"\n{type(agent).__name__}: {agent.run(question)}")
        print("阶段：" + " → ".join(item["stage"] for item in agent.trace))
    chain = ToolChain(registry, [
        ChainStep("calculator", {"expression": "{input} * 2"}, "doubled"),
        ChainStep("calculator", {"expression": "{doubled} + 1"}, "answer"),
    ])
    print("\n工具链：", chain.run(input=3))
    executor = AsyncToolExecutor(registry)
    print("并发工具：", asyncio.run(executor.run([
        ("calculator", {"expression": "1 + 2"}), ("calculator", {"expression": "sqrt(16)"}),
    ])))


if __name__ == "__main__":
    main()
