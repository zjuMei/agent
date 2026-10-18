"""组合范式：规划 → 子任务 ReAct → 汇总 → 反思修订。"""

import json

from common import make_messages, tool_catalog
from plan_solve import parse_plan
from react import run_react
from reflection import reflect_answer


MAX_TASKS = 3
TASK_STEPS = 3

PLANNER_PROMPT = """将用户请求拆成 1～3 个按顺序执行的子任务，只输出 JSON 对象数组。
本阶段只规划，不执行工具；每个子任务随后由 ReAct 执行，可观察工具结果并重试。
每项包含自由描述的 goal 和所需的 allowed_tools；只选已注册工具，不限制任务类型。
无需工具的比较、解释或澄清任务使用 allowed_tools=[]，需要多种工具时可列多个。
无工具任务不设置 finish_on_success；goal 只描述工作目标，不预写整段答案。
一次工具调用成功就足以完成的原子任务（例如更新一次偏好、查询一个城市天气），
可设 finish_on_success=true，由程序在成功后结束；多城市、需要比较或判断结果的任务设为 false。
搜索需要模型检查相关性，finish_on_success 必须为 false。中间任务不要撰写完整用户答案。
新长期偏好需要保存时单列一个子任务；根据会话历史和有效工具证据避免重复工作。
同时询问当前天气并推荐景点时，安排一个天气任务用 mode="both"，不要重复查询。
保留用户原始限制，只有结合上下文仍无法明确城市时才安排请求补充信息。
不要添加“整合最终答案”或“反思”子任务，这两步由程序统一完成。
普通问候可规划一个回应子任务；不要为了凑数增加不必要步骤。
例如：[{"goal":"查询杭州当前天气及今日预报","allowed_tools":["get_weather"],"finish_on_success":true},
       {"goal":"搜索并筛选符合本次需求的杭州景点","allowed_tools":["get_attraction"],"finish_on_success":false}]
"""

SUMMARY_PROMPT = """根据原始问题、会话上下文、子任务结果及真实工具证据，生成完整中文答案。
子任务总结不是独立事实证据。遵守本次限制，保留日期及来源，说明失败或证据不足之处。
本阶段不执行工具，也不保存偏好，直接返回完整中文答案。
"""


def run_hybrid(llm, question: str, chat_history=None) -> tuple:
    # 1. 先规划。总调用上限：1 次规划 + 3×3 次执行 + 1 次汇总 + 2 次反思/修订 = 13。
    messages = make_messages(question, chat_history)
    text = llm.ask(messages + [{"role": "system", "content": PLANNER_PROMPT + tool_catalog()}], "Hybrid：规划")
    plan = parse_plan(text, max_steps=MAX_TASKS)
    results, evidence, weather_cache = [], [], {}

    # 2. 每个子任务用有界的 ReAct 执行，并共享前面的结果与工具证据。
    for index, task in enumerate(plan, 1):
        print(f"\n=== Hybrid 子任务 {index}/{len(plan)}：{task['goal']} ===", flush=True)
        context = {"完整计划": plan, "当前步骤": task, "此前步骤结果": results, "已有工具证据": evidence}
        try:
            result, _ = run_react(
                llm, question, chat_history, max_steps=TASK_STEPS, task=context, evidence=evidence,
                allowed_tools=task["allowed_tools"], weather_cache=weather_cache,
                finish_on_success=task.get("finish_on_success", False) and bool(task["allowed_tools"])
                and "get_attraction" not in task["allowed_tools"],
            )
        except (RuntimeError, ValueError, OSError) as error:
            result = f"本子任务未完成：{error}"
            print(result, flush=True)
        results.append({"步骤": task["goal"], "结果": result})

    # 3. 统一汇总，不能在这里继续调用工具。
    context = {"完整计划": plan, "子任务结果": results, "工具证据": evidence}
    messages = make_messages(question, chat_history)
    answer = llm.ask(messages + [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ], "Hybrid：汇总")
    # 4. 只评审已有汇总稿，不再次生成初稿；最多一轮反思和修订。
    return reflect_answer(llm, question, answer, evidence, chat_history, max_iterations=1)
