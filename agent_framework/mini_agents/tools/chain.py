"""确定顺序的工具链由程序配置；独立工具可交给线程并发执行。"""

import asyncio
from dataclasses import dataclass

from ..core.exceptions import AgentError
from .registry import ToolRegistry


@dataclass(frozen=True)
class ChainStep:
    tool: str
    arguments: dict
    output_key: str


class ToolChain:
    def __init__(self, registry: ToolRegistry, steps: list[ChainStep]):
        self.registry = registry
        self.steps = steps

    def run(self, **context):
        for step in self.steps:
            arguments = {key: value.format_map(context) if isinstance(value, str) else value
                         for key, value in step.arguments.items()}
            result = self.registry.execute(step.tool, arguments)
            if not result["ok"]:
                raise AgentError(f"工具链在 {step.tool} 失败：{result['error']}")
            context[step.output_key] = result["result"]
        return context


class AsyncToolExecutor:
    """只执行调用方明确给出的独立同步工具；不自动推断依赖或重试。"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def run(self, calls: list[tuple[str, dict]]) -> list[dict]:
        return await asyncio.gather(*(
            asyncio.to_thread(self.registry.execute, name, arguments) for name, arguments in calls
        ))
