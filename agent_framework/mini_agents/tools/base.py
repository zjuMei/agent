"""类工具与普通函数工具共享同一份参数校验和 JSON Schema。"""

import inspect
import re
from abc import ABC, abstractmethod
from typing import get_type_hints

from pydantic import BaseModel, ConfigDict, create_model


class Tool(ABC):
    name: str
    description: str
    input_model: type[BaseModel]

    def schema(self):
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.input_model.model_json_schema(),
        }}

    def run(self, arguments: dict):
        values = self.input_model.model_validate(arguments, strict=True)
        return self.execute(**{name: getattr(values, name) for name in type(values).model_fields})

    @abstractmethod
    def execute(self, **arguments):
        pass


class FunctionTool(Tool):
    def __init__(self, function, *, name=None, description=None):
        if inspect.iscoroutinefunction(function):
            raise ValueError("注册同步函数；并发执行请使用 AsyncToolExecutor")
        self.function = function
        self.name = name or function.__name__
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.name):
            raise ValueError("工具名称只能包含字母、数字、下划线和连字符，最长 64 字符")
        self.description = description or inspect.getdoc(function) or self.name
        hints = get_type_hints(function, include_extras=True)
        fields = {}
        for key, parameter in inspect.signature(function).parameters.items():
            if parameter.kind not in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY):
                raise ValueError("工具参数只支持具名参数，不支持 *args、**kwargs 或仅位置参数")
            if key not in hints:
                raise ValueError(f"工具参数 {key} 缺少类型注解")
            default = ... if parameter.default is parameter.empty else parameter.default
            fields[key] = (hints[key], default)
        self.input_model = create_model(
            self.name + "Input", __config__=ConfigDict(extra="forbid", validate_default=True), **fields,
        )

    def execute(self, **arguments):
        return self.function(**arguments)
