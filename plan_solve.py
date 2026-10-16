"""Plan-and-Solve：先生成完整计划，再按固定顺序执行，不在步骤内嵌套 ReAct。"""

import json

from common import make_messages, observe, parse_json, parse_response
from tools import ActionError


PLANNER_PROMPT = """当前是规划阶段，只制定计划，不执行工具，也不声称已有查询结果。
将用户请求分解为 1～5 个按顺序执行的步骤，每步最多需要一个工具调用。
如需保存新长期偏好，将保存列为独立步骤；天气、景点搜索也分别列为独立步骤。
最后一步必须整合已有结果回答原始问题；结合对话历史仍缺少城市时才请求补充。
只输出 JSON 字符串数组，不输出 Thought、Action 或其他说明，例如：
["查询杭州今日天气", "搜索符合本次要求的杭州景点", "根据天气和来源给出最终推荐"]
"""

EXECUTOR_PROMPT = """当前是执行阶段，只完成指定步骤，不重新规划，也不跳过后续步骤。
根据原始问题、完整计划和已有结果执行。工具需要的参数只能取自用户输入或已有结果。
需要外部信息或保存偏好时输出一条 Action: 工具名(参数名="参数值")；
无需工具的中间处理输出 Result: 本步骤结果；不能假装已查询或保存。
最后一步只返回 Action: Finish[最终中文答案]，不再调用工具。
工具失败时保留失败事实，不能把失败步骤写成已完成；最终说明缺失的信息。
"""


def parse_plan(text: str, max_steps: int = 5) -> list:
    """规划模式和组合模式共用计划格式校验。"""
    plan = parse_json(text)
    if not isinstance(plan, list) or not 1 <= len(plan) <= max_steps or any(
        not isinstance(step, str) or not step.strip() for step in plan
    ):
        raise ActionError(f"计划必须是包含 1～{max_steps} 个非空步骤的 JSON 字符串数组。")
    return plan


def run_plan_solve(llm, question: str, chat_history=None) -> tuple:
    messages = make_messages(question, chat_history)
    evidence = []

    # 1. Planner 只调用一次模型，先生成整个计划。
    text = llm.ask(messages + [{"role": "system", "content": PLANNER_PROMPT}], "Plan-and-Solve：规划")
    plan = parse_plan(text)
    messages.append({"role": "assistant", "content": "完整计划：" + json.dumps(plan, ensure_ascii=False)})

    # 2. Executor 逐步调用模型；前一步的工具结果供后一步使用。
    for index, step in enumerate(plan, 1):
        last = index == len(plan)
        instruction = EXECUTOR_PROMPT + f"\n当前步骤 {index}/{len(plan)}：{step}"
        if last:
            instruction += "\n这是最后一步，必须 Finish。"
        text = llm.ask(messages + [{"role": "system", "content": instruction}], f"执行步骤 {index}：{step}")
        if text:
            messages.append({"role": "assistant", "content": text})
        try:
            kind, content = parse_response(text)
            if last:
                if kind != "finish":
                    raise ActionError("计划最后一步必须 Finish，未继续执行工具。")
                return content, evidence
            if kind == "finish":
                raise ActionError("当前为中间步骤，请勿提前 Finish；本步骤未执行工具。")
            result = observe(content) if kind == "tool" else content
            if kind == "tool":
                evidence.append(result)
        except ActionError as error:
            if last:
                raise
            result = f"本步骤失败：{error}"
            print(result, flush=True)
        messages.append({"role": "user", "content": f"步骤 {index}（{step}）的结果：{result}"})
