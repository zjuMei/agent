# Agent 学习：原生 Python、LangGraph 与自建框架

参考 [Hello-Agents 第四章](https://hello-agents.datawhale.cc/#/./chapter4/第四章%20智能体经典范式构建)。
两版均提供 ReAct、Plan-and-Solve、Reflection 和默认的 Hybrid 组合模式。

| 版本 | 执行流程 | 说明 |
| --- | --- | --- |
| [native](native/README.md) | 普通函数、循环和条件判断 | 当前原生实现，便于理解 Agent 的基本机制 |
| [langgraph_app](langgraph_app/README.md) | StateGraph 节点、条件边和子图 | 同一组功能，学习用 LangGraph 组织流程 |
| [agent_framework](agent_framework/README.md) | 自建模型、消息、工具与 Agent 接口 | 第七章框架，以及第八章的会话持久化、长期记忆和基础 RAG |

第七章自建框架的安装、运行和扩展方法见 [agent_framework/README.md](agent_framework/README.md)。
快速体验：`python -m agent_framework.examples.offline_demo`；真实模型运行：`python -m agent_framework --mode react`。
以下说明针对原生版与 LangGraph 版旅行助手。

## 安装与配置

使用当前 Conda 环境（本项目已在 Python 3.13 下验证）。原生版需要 Python 3.9+，LangGraph 版需要 Python 3.10+。
只使用原生版时安装 `native/requirements.txt`；使用两版时安装新版依赖即可：

```bash
python -m pip install -r langgraph_app/requirements.txt
```

两版共用仓库根目录 `.env`，不需要在子目录重复配置。首次使用时复制 `.env.example` 为 `.env`，已有配置不要覆盖。
`LLM_API_KEY`、`LLM_MODEL` 必填；景点搜索需要 `TAVILY_API_KEY`。
当前模型服务使用兼容 Chat Completions 的接口，需支持 Function Calling。
`LLM_ENABLE_THINKING` 通过阿里云扩展参数传入，使用其他服务商时需检查其兼容性。

统一配置默认值：关闭思考模式、temperature=0.2、max_completion_tokens=1536、非流式、请求超时 60 秒、SDK 最多重试一次。
各模式和各阶段共用同一套配置，启动时打印生效值。进程中已存在的环境变量优先于 `.env`。

## 运行

在仓库根目录执行：

```bash
python native/agent.py
python langgraph_app/agent.py

# 单次执行或选择模式
python langgraph_app/agent.py --mode react "杭州现在天气如何？"
python langgraph_app/agent.py --mode plan "杭州天气，推荐一个室内景点"
```

也可以进入对应子目录后执行 `python agent.py`。原先根目录的入口已移动到 `native/agent.py`。

两版均支持 `/mode hybrid|react|plan|reflection`、`/clear`、`/exit`。
切换模式保留当前进程的上下文；`/clear` 清空对话及工具证据，保留长期偏好。
对话只保留最近五轮，退出后不保存。

## 工具与长期偏好

- `get_weather`：当前天气、今日预报或两者；同一轮共享天气缓存。
- `get_attraction`：根据城市和要求搜索景点，返回摘要和来源。
- `save_preferences`：保存明确的长期兴趣、预算及交通偏好；临时限制不保存。

模型决定是否调用工具，程序校验权限与参数并实际执行；工具成功不等于结果一定符合用户需求。
规划步骤使用自由描述的目标和工具权限清单，不预设固定任务类型。

偏好分别保存为 `native/preferences.json` 与 `langgraph_app/preferences.json`。
新版创建时复制了原版的初始偏好，此后各自更新。两版会话也互不共享。
配置、偏好和缓存文件不会提交到 Git。

## 阅读顺序

建议先比较两版的 `react.py`，观察 `for` 循环如何变为模型节点、工具节点和条件边。
再阅读 `plan_solve.py`、`reflection.py`，最后看 `hybrid.py` 如何复用 ReAct 与评审子图。
模型封装和业务工具在两版中分别保留一份，避免跨目录导入影响理解。
