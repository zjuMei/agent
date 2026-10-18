"""默认仅注册无网络依赖的计算器；联网搜索由应用显式添加。"""

import ast
import math
import operator
import os

from pydantic import BaseModel, ConfigDict, Field

from .base import Tool
from .registry import ToolRegistry


class CalculatorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expression: str = Field(min_length=1, max_length=200, description="四则运算表达式，可用 sqrt、pi、e")


class CalculatorTool(Tool):
    name = "calculator"
    description = "计算四则运算、乘方和平方根；例如 sqrt(16) + 2 * 3"
    input_model = CalculatorInput

    def execute(self, expression):
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 80:
            raise ValueError("表达式太复杂")
        binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                  ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
                  ast.Mod: operator.mod, ast.Pow: operator.pow}

        def evaluate(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                result = node.value
            elif isinstance(node, ast.Name) and node.id in ("pi", "e"):
                result = getattr(math, node.id)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                result = evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            elif isinstance(node, ast.BinOp) and type(node.op) in binary:
                left, right = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Pow) and abs(right) > 100:
                    raise ValueError("乘方指数绝对值不能超过 100")
                result = binary[type(node.op)](left, right)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id == "sqrt" and len(node.args) == 1 and not node.keywords):
                result = math.sqrt(evaluate(node.args[0]))
            else:
                raise ValueError("只允许数值、四则运算、乘方、sqrt、pi 和 e")
            if type(result) not in (int, float) or not math.isfinite(result) or abs(result) > 1e100:
                raise ValueError("结果必须是绝对值不超过 1e100 的有限实数")
            return result

        return evaluate(tree.body)


def create_registry(*, search=False):
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    if search:
        @registry.tool
        def web_search(query: str) -> list[dict]:
            """使用 Tavily 搜索实时资料，返回标题、摘要和来源链接。"""
            from tavily import TavilyClient

            if not os.getenv("TAVILY_API_KEY"):
                raise ValueError("缺少 TAVILY_API_KEY")
            response = TavilyClient(api_key=os.environ["TAVILY_API_KEY"]).search(
                query=query, max_results=3,
            )
            return [{key: item.get(key, "") for key in ("title", "content", "url")}
                    for item in response.get("results", [])]
    return registry
