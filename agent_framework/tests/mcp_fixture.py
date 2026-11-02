"""集成测试使用的真实子进程服务，不依赖模型或网络。"""

import asyncio
import os

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

mcp = FastMCP("MCP 测试服务", log_level="ERROR")


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")
    names: list[str]
    count: int = Field(ge=1, le=3)


@mcp.tool()
def inspect_query(query: Query) -> dict:
    """验证复杂参数并回显子进程信息。"""
    return {"names": query.names, "count": query.count, "pid": os.getpid(),
            "cwd": os.getcwd(), "marker": os.environ.get("MCP_TEST_MARKER")}


@mcp.tool()
async def delayed(seconds: float) -> str:
    """等待指定秒数。"""
    await asyncio.sleep(seconds)
    return "完成"


@mcp.tool()
def broken() -> str:
    """返回工具执行错误。"""
    raise ValueError("示例工具失败")


@mcp.tool()
def disconnect() -> str:
    """模拟服务在执行中退出。"""
    os._exit(0)


if __name__ == "__main__":
    mcp.run(transport="stdio")
