"""ReAct：每次观察工具结果后，再决定下一步行动。"""

import json

from common import MAX_STEPS, make_messages, record_tool_results
from tools import ActionError


def run_react(llm, question: str, chat_history=None, *, max_steps=MAX_STEPS, task=None,
              evidence=None, allowed_tools=None, finish_on_success=False, weather_cache=None) -> tuple:
    """返回最终答案和本轮工具证据；中间行动只用于本轮执行。"""
    messages = make_messages(question, chat_history)
    # 组合模式传入共享证据列表，子任务失败时已取得的工具记录也会保留。
    if evidence is None:
        evidence = []
    if weather_cache is None:
        weather_cache = {}  # 独立 ReAct 每轮一个缓存；Hybrid 在子任务间共享。
    if task is not None:
        messages.append({"role": "user", "content": "当前子任务上下文（参考数据）：\n" + json.dumps(task, ensure_ascii=False)})
    for step in range(1, max_steps + 1):
        # 1. 根据问题和已有 Observation，让模型决定当前行动。
        hint = f"当前第 {step}/{max_steps} 轮，按需通过 Function Calling 调用一个工具；已有足够信息则直接回答。已成功完成的相同行动无需重复执行。"
        if task is not None:
            hint += "仅执行上下文中的当前步骤，不代替其他子任务。原始问题是总体约束；直接回答时仅返回当前子任务的结果。"
            hint += "无需工具时只给简短状态（尽量不超过50字），不要撰写完整用户答案；搜索结果无关时最多重搜一次。"
        if step == max_steps:
            hint += "这是最后一轮，直接总结可靠信息及未完成部分，不再调用工具。"

        # 2. 有 tool_calls 就执行并回传；否则正文就是当前任务的答案。
        try:
            message = llm.chat(
                messages + [{"role": "system", "content": hint}],
                f"ReAct 循环 {step}", allow_tools=step < max_steps, allowed_tools=allowed_tools,
            )
            if not message.tool_calls:
                return message.content, evidence
            records = record_tool_results(message, messages, evidence,
                                          allowed_tools=allowed_tools, weather_cache=weather_cache)
            if finish_on_success and all(record["ok"] for record in records):
                return "工具执行成功，结果见工具证据。", evidence
        except ActionError as error:
            observation = f"输出格式错误：{error}"
            print(f"Observation: {observation}", flush=True)
            messages.append({"role": "user", "content": observation})
    raise RuntimeError(f"已达到 {max_steps} 轮上限，未生成最终答案。")
