"""在仓库根目录运行 python -m agent_framework。"""

import argparse
import json
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from pydantic import ValidationError

from .mini_agents import (
    AgentError, Config, LLM, PlanAndSolveAgent, ReActAgent, ReflectionAgent, SimpleAgent,
    create_registry,
)


def main():
    parser = argparse.ArgumentParser(description="第七章：自己的轻量 Agent 框架")
    parser.add_argument("query", nargs="?", help="单次提问；省略时进入连续对话")
    parser.add_argument("--mode", choices=("simple", "react", "reflection", "plan"), default="react")
    parser.add_argument("--search", action="store_true", help="显式启用 Tavily 联网搜索（需额外依赖和 Key）")
    parser.add_argument("--trace", action="store_true", help="显示阶段、模型回复和工具执行记录")
    args = parser.parse_args()
    if args.search and args.mode not in ("react", "plan"):
        parser.error("只有 react 和 plan 模式使用工具")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    try:
        config = Config.from_env()
    except ValidationError as error:
        fields = ", ".join(".".join(map(str, item["loc"])) for item in error.errors())
        parser.exit(1, f"模型配置缺失或无效：{fields}。请检查根目录 .env。\n")
    except ValueError as error:
        parser.exit(1, str(error) + "\n")
    llm = LLM(config)
    classes = {"simple": SimpleAgent, "react": ReActAgent,
               "reflection": ReflectionAgent, "plan": PlanAndSolveAgent}
    options = {"tools": create_registry(search=args.search)} if args.mode in ("react", "plan") else {}
    agent = classes[args.mode](llm, **options)

    def run_turn(query):
        start, previous = perf_counter(), llm.calls
        try:
            print("\n助手：" + agent.run(query), flush=True)
            success = True
        except (AgentError, ValueError) as error:
            print(f"\n本轮失败：{error}", flush=True)
            success = False
        finally:
            if args.trace:
                print(json.dumps(agent.trace, ensure_ascii=False, indent=2), flush=True)
        print(f"本轮模型调用 {llm.calls - previous} 次，总耗时 {perf_counter() - start:.1f} 秒。", flush=True)
        return success

    try:
        print(f"模式：{args.mode}；模型：{config.model}", flush=True)
        if args.query is not None:
            if not run_turn(args.query):
                raise SystemExit(1)
            return
        print("/clear 清空会话；/exit 退出。", flush=True)
        while True:
            query = input("\n你：").strip()
            if query == "/exit":
                break
            if query == "/clear":
                agent.clear_history()
                print("已清空会话。", flush=True)
            elif query:
                run_turn(query)
    except (EOFError, KeyboardInterrupt):
        print("\n已退出。")
    finally:
        llm.close()


if __name__ == "__main__":
    main()
