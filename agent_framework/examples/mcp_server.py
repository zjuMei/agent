"""用于学习和验证的 stdio 服务；stdout 专用于 MCP，日志写 stderr。"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("示例课程服务", log_level="WARNING")


@mcp.tool()
def get_course(name: str) -> dict:
    """查询示例课程的时长；name 为课程名称。"""
    return {"name": name, "duration_minutes": 45}


@mcp.tool()
def add(a: int, b: int) -> int:
    """将两个整数相加。"""
    return a + b


if __name__ == "__main__":
    mcp.run(transport="stdio")
