"""Plan-and-Solve：先生成完整计划，再按固定顺序执行，不在步骤内嵌套 ReAct。"""

import json

from common import make_messages, parse_json, record_tool_results, tool_catalog
from tools import ActionError, TOOLS


PLANNER_PROMPT = """当前是规划阶段，只制定计划，不执行工具，也不声称已有查询结果。
将用户请求分解为 1～5 个按顺序执行的步骤，每步最多需要一个工具调用。
如需保存新长期偏好，将保存列为独立步骤；天气、景点搜索也分别列为独立步骤。
最后一步必须整合已有结果回答原始问题；结合对话历史仍缺少城市时才请求补充。
每步自由描述 goal，并在 allowed_tools 中只列本步所需的已注册工具；不需要工具时用空数组。
同时问当前天气和推荐时，一个天气步骤用 mode="both"，不拆成两个步骤。
最后一步 allowed_tools 必须为空。只输出 JSON 对象数组，例如：
[{"goal":"查询杭州今日天气","allowed_tools":["get_weather"]},
 {"goal":"搜索杭州景点","allowed_tools":["get_attraction"]},
 {"goal":"整理最终推荐","allowed_tools":[]}]
"""

EXECUTOR_PROMPT = """当前是执行阶段，只完成指定步骤，不重新规划，也不跳过后续步骤。
根据原始问题、完整计划和已有结果执行。工具需要的参数只能取自用户输入或已有结果。
需要外部信息或保存偏好时通过 Function Calling 调用一个工具；
无需工具的中间处理直接输出本步骤结果；不能假装已查询或保存。
最后一步直接返回完整中文答案，不再调用工具。
工具失败时保留失败事实，不能把失败步骤写成已完成；最终说明缺失的信息。
"""


def parse_plan(text: str, max_steps: int = 5) -> list:
    """只约束计划结构和工具权限，不预设任务类型。"""
    plan = parse_json(text)
    if not isinstance(plan, list) or not 1 <= len(plan) <= max_steps:
        raise ActionError(f"计划必须是包含 1～{max_steps} 个步骤的 JSON 对象数组。")
    for step in plan:
        if not isinstance(step, dict) or not isinstance(step.get("goal"), str) or not step["goal"].strip():
            raise ActionError("每个步骤必须有非空字符串 goal。")
        names = step.get("allowed_tools")
        if not isinstance(names, list) or any(not isinstance(name, str) or name not in TOOLS for name in names):
            raise ActionError("allowed_tools 必须是已注册工具名的数组，无需工具时用空数组。")
        if len(names) != len(set(names)):
            raise ActionError("allowed_tools 不得重复。")
        if not isinstance(step.get("finish_on_success", False), bool):
            raise ActionError("finish_on_success 必须是布尔值。")
        if step.get("finish_on_success") and (len(names) != 1 or "get_attraction" in names):
            # 这是省调用的可选提示，不是执行权限；不适用时安全降级，别中断正常任务。
            step["finish_on_success"] = False
            print("计划调整：此步骤不适用成功即结束，按正常流程执行。", flush=True)
    return plan


def run_plan_solve(llm, question: str, chat_history=None) -> tuple:
    messages = make_messages(question, chat_history)
    evidence = []

    # 1. Planner 只调用一次模型，先生成整个计划。
    text = llm.ask(messages + [{"role": "system", "content": PLANNER_PROMPT + tool_catalog()}], "Plan-and-Solve：规划")
    plan = parse_plan(text)
    if plan[-1]["allowed_tools"]:
        raise ActionError("计划最后一步必须是无需工具的最终回答。")
    messages.append({"role": "assistant", "content": "完整计划：" + json.dumps(plan, ensure_ascii=False)})
    weather_cache = {}

    # 2. Executor 逐步调用模型；前一步的工具结果供后一步使用。
    for index, step in enumerate(plan, 1):
        last = index == len(plan)
        instruction = EXECUTOR_PROMPT + f"\n当前步骤 {index}/{len(plan)}：{step['goal']}"
        if last:
            instruction += "\n这是最后一步，必须直接回答原始问题。"
        try:
            message = llm.chat(
                messages + [{"role": "system", "content": instruction}],
                f"执行步骤 {index}：{step['goal']}", allow_tools=not last, allowed_tools=step["allowed_tools"],
            )
            if last:
                return message.content, evidence
            if message.tool_calls:
                record_tool_results(message, messages, evidence,
                                    allowed_tools=step["allowed_tools"], weather_cache=weather_cache)
            else:
                messages.append({"role": "assistant", "content": message.content})
        except ActionError as error:
            if last:
                raise
            result = f"本步骤失败：{error}"
            print(result, flush=True)
            messages.append({"role": "user", "content": result})
