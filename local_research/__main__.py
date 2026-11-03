"""在仓库根目录执行 python -m local_research。"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from agent_framework.demo import StreamPrinter
from agent_framework.mini_agents import AgentError, Config, ContextConfig, LLM

from .researcher import run_research


def main():
    parser = argparse.ArgumentParser(description="使用本地 Markdown/TXT 资料生成带引用的研究报告")
    parser.add_argument("question", help="研究问题")
    parser.add_argument("--docs", default=str(Path(__file__).parent / "knowledge"), help="资料目录")
    parser.add_argument("--output", default=str(Path(__file__).parent / "workspace"), help="任务输出根目录")
    parser.add_argument("--no-stream", action="store_true", help="关闭默认流式输出")
    parser.add_argument("--trace", action="store_true", help="结束后在终端额外显示 trace（始终保存到文件）")
    parser.add_argument("--max-plan-steps", type=int, choices=range(1, 4), default=3, help="最多子问题数，默认 3")
    parser.add_argument("--max-steps", type=int, choices=range(1, 7), default=6, help="每个子问题的模型调用上限，默认 6")
    parser.add_argument("--context-chars", type=int, default=24000, help="每次模型输入字符预算，默认 24000")
    parser.add_argument("--max-output-tokens", type=int, help="可选：覆盖现有模型配置的单次输出 token 上限")
    args = parser.parse_args()
    if args.max_output_tokens is not None and args.max_output_tokens < 1:
        parser.error("--max-output-tokens 必须大于 0")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    try:
        config = Config.from_env()
        if args.max_output_tokens is not None:
            config = config.model_copy(update={"max_tokens": args.max_output_tokens})
        context = ContextConfig(max_input_chars=args.context_chars, max_memory_chars=0)
        llm = LLM(config)
        try:
            print(f"本地资料研究；模型：{config.model}；最多 {args.max_plan_steps * args.max_steps + 2} 次模型调用。",
                  flush=True)
            if not args.no_stream:
                print("以下为执行过程和报告草稿；是否通过检查以最后的任务状态为准。", flush=True)
            result = run_research(
                args.question, docs=args.docs, output=args.output, llm=llm,
                stream=not args.no_stream,
                on_event=None if args.no_stream else StreamPrinter(), context_config=context,
                max_plan_steps=args.max_plan_steps, max_steps=args.max_steps,
            )
        finally:
            llm.close()
        labels = {"completed": "已完成，引用检查通过", "insufficient": "资料不足",
                  "validation_failed": "报告检查失败，已保留草稿", "failed": "执行失败",
                  "interrupted": "已中断"}
        print(f"\n任务状态：{labels[result.status]}\n记录目录：{result.directory}", flush=True)
        if result.report_path is not None:
            print(f"报告：{result.report_path}", flush=True)
            if args.no_stream:
                print(result.report_path.read_text(encoding="utf-8"), flush=True)
        if result.error:
            print(result.error, flush=True)
        if args.trace:
            print(json.dumps(json.loads((result.directory / "trace.json").read_text(encoding="utf-8")),
                             ensure_ascii=False, indent=2), flush=True)
        return {"completed": 0, "insufficient": 2, "interrupted": 130}.get(result.status, 1)
    except ValidationError as error:
        fields = ", ".join(".".join(map(str, item["loc"])) for item in error.errors())
        print(f"模型配置缺失或无效：{fields}。请检查根目录 .env。", flush=True)
        return 1
    except (AgentError, OSError, ValueError) as error:
        print(f"启动失败：{error}", flush=True)
        return 1
    except KeyboardInterrupt:
        print("\n已中断。", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
