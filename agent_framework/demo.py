"""在仓库根目录运行 python -m agent_framework。"""

import argparse
import json
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from pydantic import ValidationError

from .mini_agents import (
    AgentError, Config, DocumentIndex, LLM, MemoryStore, PlanAndSolveAgent, ReActAgent,
    ReflectionAgent, SessionStore, SimpleAgent, SQLiteStore, create_registry,
    register_knowledge_tools,
)


class StreamPrinter:
    """接收事件并立即显示片段；不同阶段分行，避免把草稿和最终答案混在一起。"""

    def __init__(self):
        self.last_text = None

    def __call__(self, event):
        if event["type"] == "model_start":
            print(f"\n[{event['stage']}] ", end="", flush=True)
            self.last_text = None
        elif event["type"] == "text_delta":
            print(event["text"], end="", flush=True)
        elif event["type"] == "model_end":
            message = event["message"]
            if message.get("tool_calls"):
                names = ", ".join(call["function"]["name"] for call in message["tool_calls"])
                print(f"\n调用工具：{names}", flush=True)
            else:
                self.last_text = message["content"]
                print(flush=True)
        elif event["type"] == "tool_result":
            result = event["result"] if event["ok"] else event["error"]
            print(f"工具 {event['name']}（成功={event['ok']}）："
                  + json.dumps(result, ensure_ascii=False), flush=True)

    def finish(self, answer):
        # Reflection 的最后一次输出可能是评审 JSON，此时单独标明最终答案。
        if self.last_text != answer:
            print("\n助手：" + answer, flush=True)


def local_command(query, *, clear_history, memory):
    """本地命令直接执行，不让模型猜测要保存什么，也不消耗模型调用。"""
    command, _, content = query.partition(" ")
    content = content.strip()
    if command == "/clear":
        clear_history()
        print("已清空当前用户、会话和模式的聊天记录。长期记忆及文档继续保留。", flush=True)
    elif command == "/remember":
        item = memory.add(content)
        print(f"已保存记忆 {item['id']}：{item['content']}", flush=True)
    elif command == "/memories":
        items = memory.search(content, limit=10) if content else memory.list()
        print(json.dumps(items, ensure_ascii=False, indent=2), flush=True)
    elif command == "/forget":
        if not content:
            raise ValueError("用法：/forget 记忆ID")
        print("已删除记忆。" if memory.delete(content) else "未找到当前用户的这条记忆。", flush=True)
    elif command == "/update":
        memory_id, _, replacement = content.partition(" ")
        changed = memory.update(memory_id, replacement)
        print("已更新记忆。" if changed else "未找到当前用户的这条记忆。", flush=True)
    else:
        return False
    return True


AGENTS = {"simple": SimpleAgent, "react": ReActAgent,
          "reflection": ReflectionAgent, "plan": PlanAndSolveAgent}


def run_chat(args, parser, session, memory, documents):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    try:
        config = Config.from_env()
    except ValidationError as error:
        fields = ", ".join(".".join(map(str, item["loc"])) for item in error.errors())
        parser.exit(1, f"模型配置缺失或无效：{fields}。请检查根目录 .env。\n")
    except ValueError as error:
        parser.exit(1, str(error) + "\n")
    llm = LLM(config)
    try:
        options = {}
        if args.mode in ("react", "plan"):
            options["tools"] = register_knowledge_tools(
                create_registry(search=args.search), memory=memory, documents=documents,
            )
        agent = AGENTS[args.mode](llm, session=session, memory=memory, **options)
        agent.system_prompt += (
            "若有本地文档检索工具，查询本地资料时使用它；根据检索证据回答并用 [citation] 标注来源，"
            "没有匹配资料要说明，不能编造来源。持久记忆只能通过 /remember 命令明确写入，"
            "用户要求记住信息时应提示该命令，不能声称已保存。"
        )

        def run_turn(query):
            start, previous = perf_counter(), llm.calls
            printer = None if args.no_stream else StreamPrinter()
            try:
                answer = agent.run(query, stream=not args.no_stream, on_event=printer)
                if printer is None:
                    print("\n助手：" + answer, flush=True)
                else:
                    printer.finish(answer)
                success = True
            except (AgentError, ValueError) as error:
                print(f"\n本轮失败：{error}", flush=True)
                success = False
            finally:
                if args.trace:
                    print(json.dumps(agent.trace, ensure_ascii=False, indent=2), flush=True)
            print(f"本轮模型调用 {llm.calls - previous} 次，总耗时 {perf_counter() - start:.1f} 秒。", flush=True)
            return success

        print(f"模式：{args.mode}；模型：{config.model}", flush=True)
        print(f"用户：{args.user}；会话：{args.session}；已恢复 {len(agent._turns)} 轮。", flush=True)
        if args.query is not None:
            if not run_turn(args.query):
                raise SystemExit(1)
            return
        print("/clear 清空当前会话；/remember 内容；/memories [关键词]；"
              "/update 记忆ID 新内容；/forget 记忆ID；/exit 退出。", flush=True)
        while True:
            query = input("\n你：").strip()
            if query == "/exit":
                break
            if query:
                try:
                    if not local_command(query, clear_history=agent.clear_history, memory=memory):
                        run_turn(query)
                except (AgentError, ValueError) as error:
                    print(f"命令失败：{error}", flush=True)
    except (EOFError, KeyboardInterrupt):
        print("\n已退出。")
    finally:
        llm.close()


def main():
    parser = argparse.ArgumentParser(description="轻量 Agent 框架：会话、记忆与文档检索")
    parser.add_argument("query", nargs="?", help="单次提问或本地命令；省略时进入连续对话")
    parser.add_argument("--mode", choices=("simple", "react", "reflection", "plan"), default="react")
    parser.add_argument("--search", action="store_true", help="启用 Tavily 联网搜索（需额外依赖和 Key）")
    parser.add_argument("--trace", action="store_true", help="显示阶段、模型回复和工具执行记录")
    parser.add_argument("--no-stream", action="store_true", help="关闭默认流式输出")
    parser.add_argument("--db", default=str(Path(__file__).resolve().parent / "data" / "agent.db"),
                        help="SQLite 路径；:memory: 表示本次进程临时存储")
    parser.add_argument("--user", default="local", help="用户标识，隔离会话、记忆和文档")
    parser.add_argument("--session", default="default", help="会话标识，同一用户的不同模式分别保存")
    parser.add_argument("--ingest", action="append", default=[], metavar="PATH",
                        help="导入 Markdown/TXT 文件或目录，可重复指定")
    parser.add_argument("--ingest-only", action="store_true", help="只导入文档，不调用模型，无需密钥")
    args = parser.parse_args()
    if args.search and args.mode not in ("react", "plan"):
        parser.error("只有 react 和 plan 模式使用工具")
    if args.ingest_only and (not args.ingest or args.query is not None):
        parser.error("--ingest-only 需要 --ingest，且不能同时提问")
    try:
        with SQLiteStore(args.db) as db:
            memory = MemoryStore(db, user_id=args.user)
            documents = DocumentIndex(db, namespace=args.user)
            session = SessionStore(db, user_id=args.user, session_id=args.session)
            for source in args.ingest:
                path = Path(source).expanduser()
                results = documents.add_directory(path) if path.is_dir() else [documents.add_file(path)]
                print("文档导入：" + json.dumps(results, ensure_ascii=False), flush=True)
            if args.ingest_only:
                return
            if args.query is not None and local_command(
                args.query.strip(), memory=memory,
                clear_history=lambda: session.clear(AGENTS[args.mode].__name__),
            ):
                return
            run_chat(args, parser, session, memory, documents)
    except (AgentError, OSError, ValueError) as error:
        parser.exit(1, f"运行失败：{error}\n")


if __name__ == "__main__":
    main()
