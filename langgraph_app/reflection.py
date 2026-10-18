"""Reflection：先得到初稿，再独立评审并按反馈修订。"""

import json

from langgraph.graph import END, START, StateGraph

from common import WorkflowState, make_messages, parse_json, run_graph, show_node
from react import create_react_graph
from tools import ActionError


REVIEW_PROMPT = """当前是反思阶段，你是评审员，只评审答案，不执行工具或保存偏好。
对照原始问题及工具证据检查：本次限制、长期偏好、天气日期及适配性、来源支持、
地点重复、未经核实的断言。不要为了挑错而编造问题；反馈不是新的事实证据。
检查是否将“偏好”强化为“仅限”，是否把阴天直接判为不适合室外，或给烟霾擅自添加程度。
只输出 JSON 对象，例如：
{"needs_revision": true, "feedback": "第二个景点为半室外，不符合仅限室内，应删除。"}
没有实质问题时只需返回 {"needs_revision": false}；需要修订时必须给出非空 feedback。
不要输出其他文字。
"""

REFINE_PROMPT = """当前是修订阶段，根据评审反馈修改最新初稿。
只使用原始问题、已有偏好与工具证据；反馈中的新事实未经工具支持时不能直接采信。
证据不足时删除不可靠断言并说明缺失，不得编造补充，不重新查询或保存偏好。
直接返回修订后的完整中文答案。
"""


class ReviewState(WorkflowState, total=False):
    base: list
    revisions: list
    iteration: int
    needs_revision: bool
    feedback: str
    review_context: dict


def create_review_graph(llm, max_iterations=2):
    """只审阅已有答案；独立 Reflection 和 Hybrid 共用此子图。"""

    def prepare(state: ReviewState):
        return {"base": make_messages(state["question"], state.get("chat_history")),
                "revisions": [], "iteration": 0, "needs_revision": False}

    def review(state: ReviewState):
        show_node("Reflection.review")
        iteration = state["iteration"] + 1
        context = {"原始问题": state["question"], "工具证据": state["evidence"],
                   "最新初稿": state["answer"], "修订记录": state["revisions"]}
        try:
            text = llm.ask(state["base"] + [
                {"role": "system", "content": REVIEW_PROMPT},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ], f"Reflection 反思 {iteration}")
            result = parse_json(text)
            if not isinstance(result, dict) or not isinstance(result.get("needs_revision"), bool):
                raise ActionError("反思结果必须包含布尔值 needs_revision。")
            if result["needs_revision"]:
                if not isinstance(result.get("feedback"), str) or not result["feedback"].strip():
                    raise ActionError("要求修订时必须提供非空评审反馈。")
            else:
                print("评审未发现需要修改的问题，提前结束。", flush=True)
            return {"iteration": iteration, "review_context": context,
                    "needs_revision": result["needs_revision"], "feedback": result.get("feedback", "")}
        except (RuntimeError, ActionError) as error:
            print(f"反思未完成，保留最近一次完整答案：{error}", flush=True)
            return {"iteration": iteration, "needs_revision": False}

    def refine(state: ReviewState):
        show_node("Reflection.refine")
        try:
            answer = llm.ask(state["base"] + [
                {"role": "system", "content": REFINE_PROMPT},
                {"role": "user", "content": json.dumps(state["review_context"], ensure_ascii=False)},
                {"role": "user", "content": "评审反馈：" + state["feedback"]},
            ], f"Reflection 修订 {state['iteration']}")
            revisions = state["revisions"] + [{"初稿": state["answer"], "反馈": state["feedback"], "修订稿": answer}]
            if state["iteration"] >= max_iterations:
                print("已达到反思次数上限，返回最新修订稿。", flush=True)
            return {"answer": answer, "revisions": revisions}
        except (RuntimeError, ActionError) as error:
            print(f"修订未完成，保留最近一次完整答案：{error}", flush=True)
            return {"needs_revision": False}

    graph = StateGraph(ReviewState)
    graph.add_node("prepare", prepare)
    graph.add_node("review", review)
    graph.add_node("refine", refine)
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", lambda s: "review" if max_iterations > 0 else END, ["review", END])
    graph.add_conditional_edges("review", lambda s: "refine" if s["needs_revision"] else END, ["refine", END])
    graph.add_conditional_edges("refine", lambda s: "review" if s["needs_revision"] and s["iteration"] < max_iterations else END,
                                ["review", END])
    return graph.compile()


def create_reflection_graph(llm, max_iterations=2):
    graph = StateGraph(WorkflowState)
    graph.add_node("draft", create_react_graph(llm))
    graph.add_node("reflect", create_review_graph(llm, max_iterations))
    graph.add_edge(START, "draft")
    graph.add_conditional_edges("draft", lambda s: END if s.get("error") else "reflect", [END, "reflect"])
    graph.add_edge("reflect", END)
    return graph.compile()


def run_reflection(llm, question: str, max_iterations=2, chat_history=None) -> tuple:
    return run_graph(create_reflection_graph(llm, max_iterations), question, chat_history)


