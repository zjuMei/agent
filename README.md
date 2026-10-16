# 三种经典 Agent 范式与组合模式：旅行助手学习版

参考 [Hello-Agents 第四章](https://hello-agents.datawhale.cc/#/./chapter4/第四章%20智能体经典范式构建)。
网站暂时无法访问时，可阅读同一教程的 [GitHub 原文](https://github.com/datawhalechina/hello-agents/blob/main/docs/chapter4/第四章%20智能体经典范式构建.md)。

三个独立模式和默认的 `hybrid` 组合模式使用同一个模型和同一组工具，方便观察工作流程的差异。
没有引入 Agent 框架、基类或工具注册框架；每个模式有一个主函数，组合模式复用已有执行和反思函数。

## 运行

继续使用当前 Conda 环境和现有 `.env`，无需新装依赖。首次配置时：

```bash
pip install -r requirements.txt
cp .env.example .env
```

已有 `.env` 时直接使用，不需要重新复制。需要 Python 3.9+。
`LLM_API_KEY`、`LLM_MODEL` 必填，`LLM_BASE_URL` 填兼容 Chat Completions 的服务地址。
搜索景点需要 `TAVILY_API_KEY`，只问天气不需要。密钥保留在本地 `.env`。

启动连续对话（默认 Hybrid 组合模式）：

```bash
python agent.py
```

启动后可直接输入问题、连续追问，并在运行中切换范式：

```text
你：杭州天气，推荐景点
助手：……
你：只考虑室内的
助手：……
你：/mode plan
你：第二个需要预约吗？
助手：……
你：/mode reflection
你：重新检查刚才的推荐是否符合要求
助手：……
你：/clear
你：/exit
```

`/mode hybrid|react|plan|reflection` 切换模式并保留上下文；`/clear` 清空会话和工具证据，保留长期偏好。
`/exit`、`exit`、`quit` 或 Ctrl+C/Ctrl+D 退出。空输入跳过，单轮失败后仍可继续输入。
也可用 `python agent.py --mode plan` 以规划模式开始连续对话。

仍然支持一次性执行，提供问题时回答后退出：

```bash
python agent.py --mode hybrid "杭州天气，推荐仅限室内的景点"
python agent.py --mode react "杭州天气，推荐仅限室内的景点"
python agent.py --mode plan "杭州天气，推荐仅限室内的景点"
python agent.py --mode reflection "杭州天气，推荐仅限室内的景点"
```

不指定模式时使用 Hybrid：

```bash
python agent.py "杭州天气，推荐景点"
python agent.py "杭州现在的天气怎么样？"
python agent.py --help
```

## 四种模式有什么区别

| 模式 | Python 如何组织流程 | 模型主要决定什么 | 阅读入口 |
| --- | --- | --- | --- |
| `hybrid`（默认） | 规划 → 每步 ReAct → 汇总 → 反思修订 | 如何拆解、执行子任务，以及如何修正最终答案 | `hybrid.py` 的 `run_hybrid` |
| `react` | 行动 → 工具结果 → 再决定下一步，直到 Finish | 当前应该调用哪个工具，何时结束 | `react.py` 的 `run_react` |
| `plan` | 先生成完整计划，再按固定顺序逐步执行 | 如何拆解任务，以及当前步骤的行动 | `plan_solve.py` 的 `run_plan_solve` |
| `reflection` | ReAct 初稿 → 反思 → 有问题则修订，再次评审 | 初稿有什么问题，如何依据证据修正 | `reflection.py` 的 `run_reflection` |

### Hybrid：组合三个范式

```text
用户问题 → 规划子任务 → 每个子任务运行 ReAct → 汇总答案 → 反思 → 必要时修订
```

最多规划 3 个子任务，每个子任务最多 3 轮 ReAct，最后一轮只输出该子任务的结果。
每一步都能看到原始问题、会话、完整计划、前面子任务的结果和共享工具证据。
单个子任务失败时保留已经取得的工具证据，并将失败传给后续步骤及汇总阶段。
汇总和反思不调用工具；反思复用已有汇总稿，不再运行一次完整 ReAct。

组合模式最多调用模型 13 次：1 次规划 + 3×3 次子任务执行 + 1 次汇总 + 1 次评审 + 最多 1 次修订。
实际次数通常更少；评审无需修订时直接返回汇总稿。它比单独 ReAct 有更多阶段，简单问题也可以手动切换 `/mode react`。

### ReAct：边执行边决定

```text
用户问题 → 模型选择行动 → Python 调用工具 → Observation
                  ↑_____________________________|
                  └→ Finish → 最终答案
```

每轮模型输出简短的 Thought 行动摘要和 Action，例如：

```text
Thought: 先查杭州今日天气。
Action: get_weather(city="杭州", mode="today")
```

Python 把真实工具返回作为 Observation 放回消息列表。下一轮模型根据结果继续选择。
最多 6 轮，最后一轮只允许 Finish。兼容多行的 `Action: Finish[...]` 和直接的 `Finish[...]`。
空输出或格式错误会作为反馈送回模型；格式错误也计入轮数。

### Plan-and-Solve：先规划再执行

```text
用户问题 → Planner 生成步骤数组 → Executor 按顺序执行每一步 → 最后一步 Finish
```

Planner 先输出 JSON 字符串数组，例如：

```json
["查询杭州今日天气", "搜索杭州室内景点", "根据天气与来源整理最终推荐"]
```

每个执行步骤都能看到原始问题、完整计划和此前结果。
需要工具时输出 `Action: 工具调用`；只需加工已有数据的中间步骤输出 `Result: 步骤结果`。
最后一步输出 `Action: Finish[最终答案]`。

为了突出与 ReAct 的区别，这里固定计划，不加入自动重规划或步骤内的 ReAct 循环。
中间步骤失败会记录失败并传给后续步骤，最终如实说明缺失；不会默认把失败当成成功。
计划最多 5 步，所以最多 1 次规划 + 5 次执行模型调用。

### Reflection：先有答案，再评审修订

```text
用户问题 → ReAct 初稿 → Reviewer 反思
                          ├→ 无实质问题 → 返回当前答案
                          └→ 有问题 → Refiner 修订 → 再评审
```

评审员输出 JSON，例如：

```json
{"needs_revision": true, "feedback": "第二个景点包含室外动线，不符合仅限室内，应删除。"}
```

无需修订时允许只返回 `{"needs_revision": false}`；只有要求修订时才必须提供非空 `feedback`。
修订者拿到原始问题、工具证据、初稿和反馈，再生成完整答案。
评审与修订阶段不会执行工具，也不会保存新偏好；证据不足时删除不可靠内容并说明。
这些阶段仍使用同一个模型，只更换提示词，不需要额外模型或 API Key。

最多 2 轮反思，每轮最多新增 2 次模型调用；包含 ReAct 初稿在内最多 10 次调用。
评审通过时提前结束；反思或修订失败时保留最近一次完整答案。
达到次数上限只表示流程停止，不表示答案已被验证正确。模型自我评审不能代替独立事实核验。

## 建议阅读顺序

1. `agent.py`：命令行如何选择范式，拿到答案后如何输出。
2. `react.py`：先看最小的“行动—观察”循环。
3. `plan_solve.py`：对比规划与执行的分离。
4. `reflection.py`：理解初稿、反馈与修订之间的数据流。
5. `common.py`：共用模型调用、提示词、格式解析和异常反馈。
6. `tools.py`：天气、搜索、偏好存储，以及不使用 eval 的安全工具调用。
7. `hybrid.py`：最后看如何把三个阶段组合；它复用 `run_react` 和 `reflect_answer`。

这里仍采用教程风格的文本 Action 协议，没有改为原生 Function Calling。
模型生成工具名和参数，Python 检查并执行；模型自己不会直接执行 Python 函数。
四个模式的主函数都接收一个带有 `ask(messages, stage)` 方法的模型对象，以及可选的 `chat_history`，统一返回“最终答案、当前轮的工具证据”。
`agent.py` 保存最近 5 轮的用户问题、最终回答和带执行时间的工具证据；`make_messages` 在每轮组装历史对话，并重新读取长期偏好。
三种范式的行动、计划和反思草稿只在当前任务内使用，不全部带入下一轮；Reflection 的评审和修订也能看到会话上下文。
复用历史证据需考虑时间、预报日期及本次限制，缺少证据时应重新查询。失败轮只记录问题与失败提示，不把未取得的结果当作证据。
会话只保存在内存中，退出后清空，不保存到磁盘；每条新消息会重新计算当前范式的内部轮次。

## 保留的工具与长期记忆

- `get_weather(city, mode="today")`：默认读取今日最高/最低温及分时降雨预报；`current` 读取当前观测。观测时间兼容 `localObsDateTime` 和 `observation_time`，后者不擅自当作北京时间。今日按用户的上海时区匹配日期，接口缺少对应日期时明确报错。
- `get_attraction(city, requirements="", query="")`：简短关键词搜索，返回最多 3 条结果及来源，每条摘要最多 1000 字符。可选空参数有效，接口结构异常与正常空结果分别反馈。
- `save_preferences(likes, budget, transport)`：仅提供要更新的字段。明确的长期偏好自动保存，本次临时限制不长期保存；内容不变时不重复写文件。

四个模式共用脚本目录下的 `preferences.json`，每轮读取已有偏好。
Reflection 的初稿、反馈和修订记录只保存在本次运行内存中，与长期偏好分开。
例如：

```bash
python agent.py --mode react "我平时喜欢历史文化景点，通常乘地铁。"
python agent.py --mode plan "杭州天气，推荐景点"
```

只问天气时不推荐景点；本次要求优先于历史偏好。推荐数量、室内限制、去重和来源支持等由共用提示词约束，内容质量仍取决于模型。

## 模型调用与超时

在 `.env` 中可选设置 `LLM_TIMEOUT=120`，不设置时默认 120 秒。SDK 最多自动重试 1 次。
各模式的调用上限不包含 SDK 在同一次逻辑调用内部进行的重试。
每次调用显示阶段、耗时和完整模型输出；异常会显示异常类型及底层异常类型，不打印密钥。
天气、搜索、模型服务都需要网络访问，运行时间和费用随模式及模型响应而变化。
