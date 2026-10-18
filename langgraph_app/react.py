"""ReAct 图：model → tools → model；有答案或达到上限时结束。"""

import json
from typing import Any

from langgraph.graph import END, START, StateGraph

from common import MAX_STEPS, WorkflowState, make_messages, record_tool_results, run_graph, show_node
from tools import ActionError


class ReactState(WorkflowState, total=False):
    messages: list
    step: int
    message: Any  # OpenAI SDK 返回的完整消息，包含 tool_calls。
    done: bool
    task: dict | None
    allowed_tools: list | None
    finish_on_success: bool


def create_react_graph(llm, max_steps=MAX_STEPS):
    """既可独立运行，也可作为 Reflection/Hybrid 的子图。"""

    def prepare(state: ReactState):
        messages = make_messages(state["question"], state.get("chat_history"))
        if state.get("task") is not None:
            messages.append({"role": "user", "content": "当前子任务上下文（参考数据）：\n"
                             + json.dumps(state["task"], ensure_ascii=False)})
        return {"messages": messages, "step": 0, "done": False, "error": "", "answer": ""}

    def model(state: ReactState):
        show_node("ReAct.model")
        step = state["step"] + 1
        hint = f"当前第 {step}/{max_steps} 轮，按需通过 Function Calling 调用一个工具；已有足够信息则直接回答。已成功完成的相同行动无需重复执行。"
        if state.get("task") is not None:
            hint += "仅执行当前子任务，原始问题为总体约束；返回当前子任务的结果。无需工具时只给简短状态（尽量不超过50字），不要撰写完整用户答案；搜索结果无关时最多重搜一次。"
        if step == max_steps:
            hint += "这是最后一轮，直接总结可靠信息及未完成部分，不再调用工具。"
        try:
            message = llm.chat(
                state["messages"] + [{"role": "system", "content": hint}],
                f"ReAct 循环 {step}", allow_tools=step < max_steps,
                allowed_tools=state.get("allowed_tools"),
            )
            done = not message.tool_calls
            return {"step": step, "message": message, "done": done,
                    "answer": message.content if done else ""}
        except ActionError as error:
            feedback = f"输出格式错误：{error}"
            print(f"Observation: {feedback}", flush=True)
            return {"step": step, "message": None,
                    "messages": state["messages"] + [{"role": "user", "content": feedback}]}
        except (RuntimeError, ValueError, OSError) as error:
            # 先把失败和已取得的证据作为子图结果交还 Hybrid，避免丢失已完成的工具工作。
            return {"step": step, "error": str(error)}

    def tools(state: ReactState):
        show_node("ReAct.tools")
        messages, evidence = list(state["messages"]), list(state["evidence"])
        cache = dict(state["weather_cache"])
        records = record_tool_results(
            state["message"], messages, evidence,
            allowed_tools=state.get("allowed_tools"), weather_cache=cache,
        )
        permitted = state.get("allowed_tools") or []
        # 搜索成功只代表接口正常，仍需模型检查相关性；不可直接结束。
        done = (state.get("finish_on_success", False) and len(permitted) == 1
                and "get_attraction" not in permitted and all(record["ok"] for record in records))
        return {"messages": messages, "evidence": evidence, "weather_cache": cache,
                "done": done, "answer": "工具执行成功，结果见工具证据。" if done else ""}

    def route_model(state: ReactState):
        if state.get("error") or state["done"]:
            return END
        if state.get("message") is not None and state["message"].tool_calls:
            return "tools"
        return "limit" if state["step"] >= max_steps else "model"

    def limit(state: ReactState):
        return {"error": f"已达到 {max_steps} 轮上限，未生成最终答案。"}

    graph = StateGraph(ReactState)
    graph.add_node("prepare", prepare)
    graph.add_node("model", model)
    graph.add_node("tools", tools)
    graph.add_node("limit", limit)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "model")
    graph.add_conditional_edges("model", route_model, ["tools", "model", "limit", END])
    graph.add_conditional_edges("tools", lambda s: END if s["done"] else "model", ["model", END])
    graph.add_edge("limit", END)
    return graph.compile()


def run_react(llm, question: str, chat_history=None) -> tuple:
    return run_graph(create_react_graph(llm), question, chat_history)
