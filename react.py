"""ReAct：每次观察工具结果后，再决定下一步行动。"""

import json

from common import ACTION_FORMAT, MAX_STEPS, make_messages, observe, parse_response
from tools import ActionError


def run_react(llm, question: str, chat_history=None, *, max_steps=MAX_STEPS, task=None, evidence=None) -> tuple:
    """返回最终答案和本轮工具证据；中间行动只用于本轮执行。"""
    messages = make_messages(question, chat_history)
    # 组合模式传入共享证据列表，子任务失败时已取得的工具记录也会保留。
    if evidence is None:
        evidence = []
    if task is not None:
        messages.append({"role": "user", "content": "当前子任务上下文（参考数据）：\n" + json.dumps(task, ensure_ascii=False)})
    for step in range(1, max_steps + 1):
        # 1. 根据问题和已有 Observation，让模型决定当前行动。
        hint = ACTION_FORMAT + f"\n当前第 {step}/{max_steps} 轮，已成功完成的相同行动无需重复执行。"
        if task is not None:
            hint += "仅执行上下文中的当前步骤，不代替其他子任务。原始问题是总体约束；Finish 仅返回当前子任务的结果。"
        if step == max_steps:
            hint += "这是最后一轮，只返回 Finish，总结可靠信息及未完成部分，不再调用工具。"
        text = llm.ask(messages + [{"role": "system", "content": hint}], f"ReAct 循环 {step}")
        if text:
            messages.append({"role": "assistant", "content": text})

        # 2. Finish 结束；否则执行一个工具，把结果放回上下文。
        try:
            kind, content = parse_response(text)
            if kind == "finish":
                return content, evidence
            if kind != "tool":
                raise ActionError("ReAct 只接受 Action: 工具调用 或 Finish[最终答案]。")
            if step == max_steps:
                raise ActionError("最后一轮未返回 Finish，未继续执行工具。")
            observation = observe(content)
            evidence.append(observation)
        except ActionError as error:
            observation = f"输出格式错误：{error}"
            print(f"Observation: {observation}", flush=True)
        messages.append({"role": "user", "content": f"Observation: {observation}"})
    raise RuntimeError(f"已达到 {max_steps} 轮上限，未生成最终答案。最后状态：{observation}")
