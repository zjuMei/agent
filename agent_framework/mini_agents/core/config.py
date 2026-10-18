"""只读取配置；加载哪个 .env 由应用入口决定。"""

import os

from pydantic import BaseModel, Field


class Config(BaseModel):
    model: str = Field(min_length=1)
    api_key: str = Field(min_length=1, repr=False)
    base_url: str | None = None
    timeout: float = Field(default=60, gt=0, allow_inf_nan=False)
    temperature: float = Field(default=0.2, ge=0, lt=2, allow_inf_nan=False)
    max_tokens: int = Field(default=1536, gt=0)
    # 非标准参数，只在显式配置时发送给兼容服务。
    enable_thinking: bool | None = None

    @classmethod
    def from_env(cls):
        values = {
            "model": os.getenv("LLM_MODEL") or os.getenv("LLM_MODEL_ID") or "",
            "api_key": os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
            "base_url": os.getenv("LLM_BASE_URL") or None,
            "timeout": os.getenv("LLM_TIMEOUT", "60"),
            "temperature": os.getenv("LLM_TEMPERATURE", "0.2"),
            "max_tokens": os.getenv("LLM_MAX_COMPLETION_TOKENS", "1536"),
        }
        thinking = os.getenv("LLM_ENABLE_THINKING")
        if thinking is not None:
            if thinking.lower() not in ("true", "false"):
                raise ValueError("LLM_ENABLE_THINKING 必须为 true 或 false")
            values["enable_thinking"] = thinking.lower() == "true"
        return cls(**values)
