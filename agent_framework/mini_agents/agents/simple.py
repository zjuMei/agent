from ..core.agent import Agent


class SimpleAgent(Agent):
    """最小的有上下文对话：用户消息 → 模型 → 助手消息。"""

    def _run(self, messages):
        reply = self._chat(messages)
        messages.append(reply)
        return reply.content
