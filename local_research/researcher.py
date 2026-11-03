"""应用编排：每次研究有独立索引、证据和输出目录，框架核心保持不变。"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from agent_framework.mini_agents import ContextBuilder, ContextConfig, DocumentIndex, PlanAndSolveAgent, SQLiteStore

from .prompts import SYSTEM_PROMPT
from .report import append_sources, insufficient_report, validate_report, write_json
from .tools import ResearchTools


@dataclass(frozen=True)
class ResearchResult:
    task_id: str
    directory: Path
    status: str
    report_path: Path | None
    error: str | None


def _now():
    return datetime.now(timezone.utc).isoformat()


def _files(root, output):
    seen = set()
    for file in sorted(root.rglob("*")):
        resolved = file.resolve()
        if (file.is_file() and file.suffix.lower() in (".md", ".txt")
                and resolved.is_relative_to(root) and not resolved.is_relative_to(output)
                and resolved not in seen):
            seen.add(resolved)
            yield resolved


def run_research(question: str, *, docs, output, llm, stream=True, on_event=None,
                 max_plan_steps=3, max_steps=6, context_config=None) -> ResearchResult:
    """llm 由调用方提供并关闭；执行失败保留记录，返回 failed/validation_failed 状态。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("研究问题不能为空")
    if type(max_plan_steps) is not int or not 1 <= max_plan_steps <= 3:
        raise ValueError("第一版研究最多拆分 1 至 3 个子问题")
    if type(max_steps) is not int or not 1 <= max_steps <= 6:
        raise ValueError("每个子问题最多进行 1 至 6 次模型调用")
    context = ContextBuilder(context_config if context_config is not None else ContextConfig(max_memory_chars=0))
    root, output = Path(docs).expanduser().resolve(), Path(output).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"资料目录不存在：{root}")
    if root.is_relative_to(output):
        raise ValueError("资料目录不能位于输出目录内；请使用独立的输出目录")
    task_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
    directory = output / task_id
    directory.mkdir(parents=True, exist_ok=False)
    task = {"task_id": task_id, "question": question.strip(), "docs": str(root),
            "status": "running", "started_at": _now(), "imports": [], "plan": [],
            "model_attempts": 0, "validation_errors": [],
            "settings": {"max_plan_steps": max_plan_steps, "max_steps": max_steps,
                         "max_input_chars": context.config.max_input_chars,
                         "max_tool_result_chars": context.config.max_tool_result_chars}}
    model_config = getattr(llm, "config", None)
    task["model"] = getattr(model_config, "model", type(llm).__name__)
    task["settings"].update(stream=stream, max_output_tokens=getattr(model_config, "max_tokens", None))
    write_json(directory / "task.json", task)
    started, agent, research_tools, report_path, error = perf_counter(), None, None, None, None

    def observe(event):
        if event["type"] == "model_start":
            task["model_attempts"] += 1
        if on_event is not None:
            on_event(event)

    try:
        with SQLiteStore(directory / "index.sqlite3") as db:
            documents = DocumentIndex(db, namespace=task_id)
            for file in _files(root, output):
                imported = documents.add_file(file)
                task["imports"].append({**imported, "source": file.relative_to(root).as_posix()})
            research_tools = ResearchTools(documents, root)
            if any(item["chunks"] for item in task["imports"]):
                agent = PlanAndSolveAgent(llm, name="本地资料研究助手", tools=research_tools.registry,
                                          system_prompt=SYSTEM_PROMPT, max_plan_steps=max_plan_steps,
                                          max_steps=max_steps, context=context)
                draft = agent.run(question.strip(), stream=stream, on_event=observe)
                evidence = research_tools.snapshot()
                if not any(item["pages"] for item in evidence):
                    task["status"] = "insufficient"
                    write_draft = directory / "draft.md"
                    write_draft.write_text(draft, encoding="utf-8")
                    report = insufficient_report(question.strip())
                else:
                    errors, used = validate_report(draft, evidence)
                    task["validation_errors"] = errors
                    if errors:
                        task["status"] = "validation_failed"
                        error = "引用或报告结构检查失败：" + "；".join(errors)
                        (directory / "draft.md").write_text(draft, encoding="utf-8")
                        report = None
                    else:
                        task["status"] = "completed"
                        report = append_sources(draft, evidence, used)
            else:
                task["status"] = "insufficient"
                report = insufficient_report(question.strip())
            if report is not None:
                report_path = directory / "report.md"
                report_path.write_text(report, encoding="utf-8")
    except KeyboardInterrupt:
        task["status"], error = "interrupted", "用户中断本次研究；未生成完成报告"
    except Exception as failure:
        task["status"] = "failed"
        error = f"{type(failure).__name__}: {failure}"
    finally:
        trace = agent.trace if agent is not None else []
        task.update(finished_at=_now(), seconds=round(perf_counter() - started, 3), error=error,
                    report=report_path.name if report_path is not None else None)
        task["plan"] = next((entry["steps"] for entry in trace if entry.get("stage") == "计划"), [])
        write_json(directory / "evidence.json", research_tools.snapshot() if research_tools else [])
        write_json(directory / "trace.json", trace)
        write_json(directory / "task.json", task)
    return ResearchResult(task_id, directory, task["status"], report_path, error)
