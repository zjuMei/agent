# 三种经典 Agent 范式与组合模式：旅行助手学习版

参考 [Hello-Agents 第四章](https://hello-agents.datawhale.cc/#/./chapter4/第四章%20智能体经典范式构建)。
网站暂时无法访问时，可阅读同一教程的 [GitHub 原文](https://github.com/datawhalechina/hello-agents/blob/main/docs/chapter4/第四章%20智能体经典范式构建.md)。

三个独立模式和默认的 `hybrid` 组合模式使用同一个模型和同一组工具，方便观察工作流程的差异。
没有引入 Agent 框架、基类或工具注册框架；每个模式有一个主函数，组合模式复用已有执行和反思函数。
工具调用统一使用 [Function Calling](https://developers.openai.com/api/docs/guides/function-calling)：模型返回结构化调用，Python 校验参数并执行。

## 运行

以下命令在 `native/` 目录执行。继续使用当前 Conda 环境；配置读取仓库根目录的 `.env`。
首次配置时执行（已有 `.env` 时不要覆盖）：

```bash
pip install -r requirements.txt
cp ../.env.example ../.env
```

已有 `.env` 时直接使用，不需要重新复制。需要 Python 3.9+。
从仓库根目录也可执行 `python native/agent.py`，不受启动目录影响。
`LLM_API_KEY`、`LLM_MODEL` 必填，`LLM_BASE_URL` 填兼容 Chat Completions 的服务地址。
服务与模型还需支持 `tools`、`tool_choice`、`parallel_tool_calls` 和工具结果回传；仅支持普通聊天不够。
工具 Schema 使用 `strict: true`，严格模式的支持程度取决于服务商；本地仍校验工具名、参数名和类型。
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
| `react` | 行动 → 工具结果 → 再决定下一步，直到直接回答 | 当前应该调用哪个工具，何时结束 | `react.py` 的 `run_react` |
| `plan` | 先生成完整计划，再按固定顺序逐步执行 | 如何拆解任务，以及当前步骤的行动 | `plan_solve.py` 的 `run_plan_solve` |
| `reflection` | ReAct 初稿 → 反思 → 有问题则修订，再次评审 | 初稿有什么问题，如何依据证据修正 | `reflection.py` 的 `run_reflection` |

### Hybrid：组合三个范式

```text
用户问题 → 规划子任务 → 每个子任务运行 ReAct → 汇总答案 → 反思 → 必要时修订
```

最多规划 3 个子任务，每个子任务最多 3 轮 ReAct，最后一轮只输出该子任务的结果。
计划不使用固定任务类型，而是让模型自由填写目标及本步可用工具，例如：

```json
[
  {"goal": "更新长期偏好", "allowed_tools": ["save_preferences"], "finish_on_success": true},
  {"goal": "查询杭州当前天气和今日预报", "allowed_tools": ["get_weather"], "finish_on_success": true},
  {"goal": "搜索并筛选一个杭州景点", "allowed_tools": ["get_attraction"], "finish_on_success": false}
]
```

`allowed_tools=[]` 表示只分析已有资料；任务需要多种工具时可列多个。
程序只向模型提供本步获准工具，执行前再次校验；未知工具名或非法计划会报错。
工具权限限制执行范围，不保证模型的规划和分工一定正确；新增外部能力仍需注册相应函数和 Schema。
`finish_on_success` 可省略，默认 false。只有一次工具调用就足以完成的原子任务才设为 true，
例如更新一次偏好或获取一个城市所需天气；多城市查询、比较和筛选仍需循环。
无工具、多工具或搜索步骤误设为 true 时，程序会将其降为 false，继续正常执行；
该标记只是节省调用的优化提示。未知工具、非法权限结构等实际错误仍会拦截。
工具明确返回 `ok=true` 后程序才结束这类子任务，无需模型再总结。失败仍反馈给模型，受原轮数限制。
景点搜索始终保留观察结果的步骤，允许最多一次改写关键词重搜；完成时只输出简短状态，完整答案由汇总阶段生成。
每一步都能看到原始问题、会话、完整计划、前面子任务的结果和共享工具证据。
单个子任务失败时保留已经取得的工具证据，并将失败传给后续步骤及汇总阶段。
汇总和反思不调用工具；反思复用已有汇总稿，不再运行一次完整 ReAct。

组合模式最多调用模型 13 次：1 次规划 + 3×3 次子任务执行 + 1 次汇总 + 1 次评审 + 最多 1 次修订。
实际次数通常更少；评审无需修订时直接返回汇总稿。它比单独 ReAct 有更多阶段，简单问题也可以手动切换 `/mode react`。
以“新偏好＋当前天气＋景点推荐”为例，正常成功时可为 7 次：规划 1 次、偏好 1 次、天气 1 次、
搜索及检查 2 次、汇总 1 次、评审 1 次；重搜、修订或失败重试会增加调用。

### ReAct：边执行边决定

```text
用户问题 → 模型返回 tool_calls → Python 调用工具 → role=tool 回传结果
                  ↑____________________________________|
                  └→ 无 tool_calls，返回正文 → 最终答案
```

每轮模型按需通过专用字段选择工具，例如以下是响应中的一个调用：

```json
{
  "id": "call_001",
  "type": "function",
  "function": {
    "name": "get_weather",
    "arguments": "{\"city\":\"杭州\",\"mode\":\"today\"}"
  }
}
```

Python 先保留 assistant 的调用，再追加 `role="tool"`、对应 `tool_call_id` 和含 `ok`、工具名、参数、时间、耗时及结果的 JSON 记录。
`ok` 由执行是否抛出异常决定，不从“失败”等文字推测；它表示工具执行成功，不保证搜索结果相关或事实正确。
下一轮模型根据结果继续选择；没有工具调用时，正文就是当前任务的答案，不再解析 `Finish[...]`。
最多 6 轮，最后一轮不提供工具，只总结已有结果。每轮通过 `parallel_tool_calls=False` 限制为一个工具，
服务仍返回多个调用时不执行，作为格式错误反馈。空输出、截断或格式错误也计入轮数。
模型可以附一句行动说明，但不要求输出 `Thought:`。日志打印工具名、JSON 参数和 Observation；没有 Thought 文本不表示模型没有推理。

### Plan-and-Solve：先规划再执行

```text
用户问题 → Planner 生成步骤数组 → Executor 按顺序执行每一步 → 最后一步直接回答
```

Planner 先输出 JSON 对象数组，例如：

```json
[
  {"goal": "查询杭州今日天气", "allowed_tools": ["get_weather"]},
  {"goal": "搜索杭州室内景点", "allowed_tools": ["get_attraction"]},
  {"goal": "根据天气与来源整理最终推荐", "allowed_tools": []}
]
```

每个执行步骤都能看到原始问题、完整计划和此前结果。
需要工具时返回 `tool_calls`；只需加工已有数据的中间步骤直接返回文字结果。
最后一步不提供工具，直接返回最终答案。

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
5. `common.py`：共用模型调用、提示词、工具消息回传和异常反馈。
6. `tools.py`：天气、搜索、偏好存储，以及工具 Schema 和 JSON 参数校验。
7. `hybrid.py`：最后看如何把三个阶段组合；它复用 `run_react` 和 `reflect_answer`。

Function Calling 只替换工具调用协议，四种范式的组织方式保持不变；不增加新的运行模式。
模型生成工具名和 JSON 参数，Python 检查并执行；模型自己不会直接执行 Python 函数。
`LLM.chat(messages, stage, allow_tools=True, allowed_tools=[...])` 返回含正文和 `tool_calls` 的完整消息，供执行阶段使用。
省略 `allowed_tools` 表示可用全部工具，空数组表示无工具；规划和组合模式按步骤收紧权限。
`LLM.ask(messages, stage)` 不提供工具，只返回正文，供规划、评审、汇总和修订使用。
四个模式的主函数都接收这个模型对象及可选的 `chat_history`，统一返回“最终答案、当前轮的工具证据”。
`agent.py` 保存最近 5 轮的用户问题、最终回答和带执行时间的工具证据；`make_messages` 在每轮组装历史对话，并重新读取长期偏好。
三种范式的行动、计划和反思草稿只在当前任务内使用，不全部带入下一轮；Reflection 的评审和修订也能看到会话上下文。
复用历史证据需考虑时间、预报日期及本次限制，缺少证据时应重新查询。失败轮只记录问题与失败提示，不把未取得的结果当作证据。
会话只保存在内存中，退出后清空，不保存到磁盘；每条新消息会重新计算当前范式的内部轮次。

## 保留的工具与长期记忆

- `get_weather(city, mode="today")`：默认读取今日最高/最低温及分时降雨预报；`current` 读取当前观测；`both` 一次返回两者，适合同时询问当前天气和推荐景点。观测时间兼容 `localObsDateTime` 和 `observation_time`，后者不擅自当作北京时间。今日按用户的上海时区匹配日期，接口缺少对应日期时明确报错。
- 天气成功响应按城市与日期在本轮缓存，切换 mode 可复用原始 JSON，保留原获取时间；下一轮重新创建缓存，不跨轮长期复用。失败结果不新增缓存，`both` 部分缺失时保留已取得信息并返回失败状态。
- `get_attraction(city, requirements="", query="")`：简短关键词搜索，返回最多 3 条结果及来源，每条摘要最多 600 字符。可选空参数有效，接口结构异常与正常空结果分别反馈。
- `save_preferences(likes, budget, transport)`：要更新的字段传字符串，其余传 `null`，表示保留原值；三个字段全为 `null` 时不写文件。明确的长期偏好自动保存，本次临时限制不长期保存；内容不变时不重复写文件。

原生版四个模式共用 `native/preferences.json`，每轮读取已有偏好。
LangGraph 版有独立偏好文件；两版的会话与偏好互不影响。
Reflection 的初稿、反馈和修订记录只保存在本次运行内存中，与长期偏好分开。
例如：

```bash
python agent.py --mode react "我平时喜欢历史文化景点，通常乘地铁。"
python agent.py --mode plan "杭州天气，推荐景点"
```

只问天气时不推荐景点；本次要求优先于历史偏好。推荐数量、室内限制、去重和来源支持等由共用提示词约束，内容质量仍取决于模型。
偏好用于排序，不自动变成“仅限”；合并默认偏好和新条件时不补充用户没有表达的反向条件。
条件与限制的语义判断仍依赖模型；工具权限与 JSON 格式校验不能保证这些判断正确。

## 模型调用与超时

所有模式和阶段统一使用以下配置，可在 `.env` 中覆盖，修改后重启程序生效：

```dotenv
LLM_ENABLE_THINKING=false
LLM_TEMPERATURE=0.2
LLM_MAX_COMPLETION_TOKENS=1536
LLM_TIMEOUT=60
```

默认关闭深度思考，temperature 为 0.2，完整输出上限为 1536 Token，保持非流式输出。
`enable_thinking` 是阿里云扩展，通过 `extra_body` 传入，当前配置面向正在使用的 Qwen 兼容服务。
关闭深度思考不取消 ReAct 工具选择或 Reflection 评审流程；也不保证模型判断一定正确。
程序启动时校验配置并打印生效参数，不打印密钥；不按阶段另设预算，也不静默删除不支持的参数。
输出被上限截断时仍按 `finish_reason="length"` 处理，不执行不完整工具调用或解析不完整 JSON。

请求超时默认为 60 秒，SDK 最多自动重试 1 次；超时不是整轮对话的总期限，重试后可能超过 60 秒。
各模式的调用上限不包含 SDK 在同一次逻辑调用内部进行的重试。
每次调用显示阶段、耗时和完整模型输出；异常会显示异常类型及底层异常类型，不打印密钥。
同时显示输入/输出 Token、请求 ID 和本轮累计逻辑调用次数；服务未返回的数据标为“未提供”。
输出总计可能包含未显示的推理 Token；服务提供 `reasoning_tokens` 时单独显示，否则标为“未提供”，不根据正文长度猜测。
每个工具显示成功状态与耗时；每轮结束显示模型累计耗时、总耗时及已报告 Token 合计，缺少用量时说明合计不完整。
合计不是完整计费账单，尤其不能据此推算 SDK 失败重试的 token 用量。诊断重试可使用 `OPENAI_LOG=info python agent.py`，
由 SDK 输出重试信息；默认不增加额外网络重试层。当前保持非流式输出，统计不能直接区分服务端排队、生成与网络等待。
天气、搜索、模型服务都需要网络访问，运行时间和费用随模式及模型响应而变化。
