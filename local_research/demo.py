"""无需密钥：预设模型回复，真实执行规划、SQLite、检索、阅读与引用检查。"""

import argparse
import json
from pathlib import Path

from agent_framework.demo import StreamPrinter
from agent_framework.mini_agents import Message, ToolCall

from .researcher import run_research


class DemoModel:
    """只演示固定问题的控制流，不代表真实模型的研究能力。"""

    def __init__(self):
        self.calls = 0
        self.citation = None

    def chat(self, messages, *, tools=None, stream=True, on_token=None):
        self.calls += 1
        last = messages[-1]
        if not tools:
            if "将任务分解为" in last.content:
                reply = Message("assistant", '["搜索并阅读本地资料研究流程"]')
            else:
                reply = Message("assistant", demo_report(self.citation))
        elif last.role != "tool":
            reply = Message("assistant", tool_calls=[
                ToolCall("search1", "search_documents", '{"query":"本地资料研究流程"}'),
            ])
        else:
            payload = json.loads(last.content)
            if not payload["ok"]:
                raise ValueError("离线演示工具失败：" + payload["error"])
            result = payload["result"]
            if isinstance(result, list):
                if not result:
                    raise ValueError("离线演示未找到随附资料")
                reply = Message("assistant", tool_calls=[ToolCall(
                    "read1", "read_chunk", json.dumps({"chunk_id": result[0]["chunk_id"]}),
                )])
            else:
                self.citation = result["citation"]
                reply = Message("assistant", f"已读取原文：流程包含问题拆解、检索、阅读和报告生成。[{self.citation}]")
        if stream and on_token is not None:
            for text in reply.content:
                on_token(text)
        return reply


def demo_report(citation):
    return ("# 本地资料研究流程\n\n"
            f"## 结论摘要\n\n流程包括问题拆解、检索、原文阅读与报告生成。[{citation}]\n\n"
            f"## 分问题分析\n\n先拆解研究问题，再检索并按需阅读原文，保存证据及来源。[{citation}]\n\n"
            f"## 对比与适用条件\n\n适合围绕已准备的本地资料做整理与比较。[{citation}]\n\n"
            "## 资料不足与未解决问题\n\n本演示仅使用预设回复说明控制流，不评估模型的研究质量。\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(Path(__file__).parent / "workspace"))
    args = parser.parse_args()
    print("离线控制流演示：模型回复为预设数据；索引、工具调用、证据和文件输出实际执行。", flush=True)
    result = run_research("本地资料研究流程包含哪些步骤？", docs=Path(__file__).parent / "knowledge",
                          output=args.output, llm=DemoModel(), on_event=StreamPrinter())
    print(f"\n状态：{result.status}\n结果目录：{result.directory}", flush=True)
    if result.error:
        print(result.error, flush=True)
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
