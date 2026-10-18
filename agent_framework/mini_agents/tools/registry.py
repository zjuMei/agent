"""注册表是执行工具的唯一入口；模型只能选择已注册的工具。"""

import json

from .base import FunctionTool, Tool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        if tool.name in self._tools:
            raise ValueError(f"工具已注册：{tool.name}")
        self._tools[tool.name] = tool
        return tool

    def tool(self, function=None, *, name=None, description=None):
        """@registry.tool 或 @registry.tool(name='...')，保留原函数可直接调用。"""
        def decorate(func):
            self.register(FunctionTool(func, name=name, description=description))
            return func
        return decorate(function) if function is not None else decorate

    def schemas(self):
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: str | dict) -> dict:
        try:
            if name not in self._tools:
                raise ValueError(f"未注册的工具：{name}")
            values = json.loads(arguments) if isinstance(arguments, str) else arguments
            if not isinstance(values, dict):
                raise ValueError("工具参数必须是 JSON 对象")
            result = self._tools[name].run(values)
            # 先检查可序列化性，保证结果能写回模型。函数副作用无法自动撤销。
            json.dumps(result, ensure_ascii=False, allow_nan=False)
            return {"ok": True, "result": result}
        except Exception as error:
            return {"ok": False, "error": f"{type(error).__name__}: {error}"}
