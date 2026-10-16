"""Reflection：先得到初稿，再独立评审并按反馈修订。"""

import json

from common import make_messages, parse_json, parse_response
from react import run_react
from tools import ActionError


REVIEW_PROMPT = """当前是反思阶段，你是评审员，只评审答案，不执行工具或保存偏好。
对照原始问题及工具证据检查：本次限制、长期偏好、天气日期及适配性、来源支持、
地点重复、未经核实的断言。不要为了挑错而编造问题；反馈不是新的事实证据。
只输出 JSON 对象，例如：
{"needs_revision": true, "feedback": "第二个景点为半室外，不符合仅限室内，应删除。"}
没有实质问题时只需返回 {"needs_revision": false}；需要修订时必须给出非空 feedback。
不要输出其他文字。
"""

REFINE_PROMPT = """当前是修订阶段，根据评审反馈修改最新初稿。
只使用原始问题、已有偏好与工具证据；反馈中的新事实未经工具支持时不能直接采信。
证据不足时删除不可靠断言并说明缺失，不得编造补充，不重新查询或保存偏好。
只返回 Action: Finish[修订后的完整中文答案]。
"""


def run_reflection(llm, question: str, max_iterations: int = 2, chat_history=None) -> tuple:
    # 1. 先用 ReAct 查询真实数据并得到初稿。
    answer, evidence = run_react(llm, question, chat_history)
    return reflect_answer(llm, question, answer, evidence, chat_history, max_iterations)


def reflect_answer(llm, question: str, answer: str, evidence: list, chat_history=None, max_iterations: int = 2) -> tuple:
    """只评审和修订已有答案，不重新运行 ReAct 或执行工具。"""
    # 评审、修订也能看到会话上下文；最新初稿不被当作工具证据。
    base = make_messages(question, chat_history)
    revisions = []  # 本次执行的临时记录，不写入 preferences.json。

    # 2. 同一模型通过不同提示词担任评审员和修订者。
    for iteration in range(1, max_iterations + 1):
        context = {"原始问题": question, "工具证据": evidence, "最新初稿": answer, "修订记录": revisions}
        data = {"role": "user", "content": json.dumps(context, ensure_ascii=False)}
        try:
            text = llm.ask(base + [{"role": "system", "content": REVIEW_PROMPT}, data], f"Reflection 反思 {iteration}")
            review = parse_json(text)
            if not isinstance(review, dict) or not isinstance(review.get("needs_revision"), bool):
                raise ActionError("反思结果必须包含布尔值 needs_revision。")
            if not review["needs_revision"]:
                print("评审未发现需要修改的问题，提前结束。", flush=True)
                return answer, evidence
            if not isinstance(review.get("feedback"), str) or not review["feedback"].strip():
                raise ActionError("要求修订时必须提供非空评审反馈。")
            feedback = {"role": "user", "content": "评审反馈：" + review["feedback"]}
            text = llm.ask(base + [{"role": "system", "content": REFINE_PROMPT}, data, feedback], f"Reflection 修订 {iteration}")
            kind, refined = parse_response(text)
            if kind != "finish":
                raise ActionError("修订阶段必须 Finish，未执行任何工具。")
            revisions.append({"初稿": answer, "反馈": review["feedback"], "修订稿": refined})
            answer = refined
        except (RuntimeError, ActionError) as error:
            print(f"反思或修订未完成，保留最近一次完整答案：{error}", flush=True)
            return answer, evidence
    print("已达到反思次数上限，返回最新修订稿。", flush=True)
    return answer, evidence
