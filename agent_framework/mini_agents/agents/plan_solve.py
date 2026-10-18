import json

from ..core.exceptions import OutputError
from ..core.message import Message
from .react import ReActAgent
from .reflection import parse_json


class PlanAndSolveAgent(ReActAgent):
    """先规划，再让 ReAct 逐步执行，最后汇总；规划/汇总阶段不给工具。"""

    def __init__(self, llm, *, max_plan_steps=5, **kwargs):
        super().__init__(llm, **kwargs)
        if max_plan_steps < 1:
            raise ValueError("max_plan_steps 必须大于 0")
        self.max_plan_steps = max_plan_steps

    def _run(self, messages):
        catalog = json.dumps(self.tools.schemas(), ensure_ascii=False)
        raw = self._text([*messages, Message("user",
            f"将任务分解为 1 至 {self.max_plan_steps} 个有序步骤。只返回 JSON 字符串数组，"
            f"例如 [\"第一步\", \"第二步\"]。执行阶段可用工具：{catalog}"),
        ], stage="规划")
        plan = parse_json(raw)
        if (not isinstance(plan, list) or not 1 <= len(plan) <= self.max_plan_steps
                or any(not isinstance(step, str) or not step.strip() for step in plan)):
            raise OutputError("计划必须是长度在限制内的非空字符串数组")
        self.trace.append({"stage": "计划", "steps": plan})
        work = [*messages, Message("assistant", "计划：" + json.dumps(plan, ensure_ascii=False))]
        for index, step in enumerate(plan, 1):
            work.append(Message("user", f"执行步骤 {index}：{step}。结合已有结果，只回答当前步骤。"))
            self._tool_loop(work, stage=f"步骤 {index}")
        answer = self._text([*work, Message("user",
            "现在根据各步骤结果回答最初的问题。失败或缺失的工具证据要如实说明。"),
        ], stage="汇总")
        # 内部规划和步骤请求留在 trace 中，跨轮只保存原始问答。
        messages.append(Message("assistant", answer))
        return answer
