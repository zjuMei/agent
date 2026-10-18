"""学习入口：同一个旅行问题，选择不同的 Agent 工作流程。"""

import argparse
import os
from pathlib import Path
from time import perf_counter

try:
    import readline  # 启用 input() 的行编辑和重绘，避免中文退格后显示残留。
except ImportError:
    pass  # 没有 readline 的平台仍可使用普通 input()。

from dotenv import load_dotenv

from common import LLM
from hybrid import run_hybrid
from plan_solve import run_plan_solve
from react import run_react
from reflection import run_reflection


RUNNERS = {"hybrid": run_hybrid, "react": run_react, "plan": run_plan_solve, "reflection": run_reflection}
MAX_TURNS = 5
COMMANDS = "/mode hybrid|react|plan|reflection 切换模式；/clear 清空对话；/exit 退出"


def run_turn(llm, question: str, mode: str, chat_history: list) -> bool:
    """一条用户消息触发一次范式流程，只留下问答和工具证据。"""
    print(f"\n运行模式: {mode}\n用户输入: {question}\n" + "=" * 40, flush=True)
    llm.reset_stats()
    start = perf_counter()
    success = True
    try:
        answer, evidence = RUNNERS[mode](llm, question, chat_history=chat_history)
    except (RuntimeError, ValueError, OSError) as error:
        answer, evidence = f"本轮处理失败：{error}", []
        success = False
    chat_history.append({"question": question, "answer": answer, "evidence": evidence})
    del chat_history[:-MAX_TURNS]
    print(f"\n助手：{answer}", flush=True)
    print(
        f"\n本轮统计：模型调用 {llm.calls} 次，模型累计 {llm.model_seconds:.1f} 秒，"
        f"总耗时 {perf_counter() - start:.1f} 秒；已报告 Token："
        f"输入 {llm.input_tokens}，输出 {llm.output_tokens}。",
        flush=True,
    )
    if llm.missing_usage:
        print(f"其中 {llm.missing_usage} 次调用缺少完整用量，Token 合计不完整。", flush=True)
    return success


def main():
    parser = argparse.ArgumentParser(description="学习 ReAct、Plan-and-Solve、Reflection 及组合模式")
    parser.add_argument("query", nargs="?", help="省略时进入连续对话；提供时只处理这一个问题")
    parser.add_argument("--mode", choices=tuple(RUNNERS), default="hybrid", help="默认 hybrid；plan 为 Plan-and-Solve")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    for name in ("LLM_API_KEY", "LLM_MODEL"):
        if not os.getenv(name):
            raise SystemExit(f"缺少配置 {name}，请参照 .env.example 配置 .env。")

    try:
        llm = LLM()
    except (RuntimeError, ValueError, OSError) as error:
        raise SystemExit(str(error)) from None
    chat_history = []
    mode = args.mode
    if args.query is not None:
        if not run_turn(llm, args.query, mode, chat_history):
            raise SystemExit(1)
        return

    print(f"连续对话已启动，当前模式：{mode}。\n{COMMANDS}", flush=True)
    try:
        while True:
            question = input("\n你：").strip()
            if not question:
                continue
            if question.lower() in ("/exit", "exit", "quit"):
                break
            if question == "/clear":
                chat_history.clear()
                print("已清空当前对话和工具证据，长期偏好保留。", flush=True)
                continue
            if question.split()[0] == "/mode":
                parts = question.split()
                if len(parts) == 2 and parts[1] in RUNNERS:
                    mode = parts[1]
                    print(f"已切换为 {mode}，保留当前对话上下文。", flush=True)
                else:
                    print(COMMANDS, flush=True)
                continue
            if question.startswith("/"):
                print(COMMANDS, flush=True)
                continue
            run_turn(llm, question, mode, chat_history)
    except (EOFError, KeyboardInterrupt):
        pass
    print("\n已退出。", flush=True)


if __name__ == "__main__":
    main()
