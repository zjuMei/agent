"""共用工具、Function Calling 定义，以及工具参数校验。"""

import inspect
import json
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import requests
from tavily import TavilyClient


MEMORY_FILE = Path(__file__).with_name("preferences.json")


class ActionError(ValueError):
    """模型行动格式或参数不正确。"""


class ToolError(RuntimeError):
    """工具配置或接口数据不正确。"""


def load_preferences() -> dict:
    if not MEMORY_FILE.exists():
        return {}
    preferences = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    if not isinstance(preferences, dict) or any(
        key not in ("likes", "budget", "transport") or not isinstance(value, str)
        for key, value in preferences.items()
    ):
        raise ValueError("偏好文件应为 likes、budget、transport 字符串组成的 JSON 对象")
    return preferences


def save_preferences(likes=None, budget=None, transport=None) -> str:
    preferences = load_preferences()
    updates = {"likes": likes, "budget": budget, "transport": transport}
    updates = {key: value for key, value in updates.items() if value is not None}
    if not updates:
        raise ActionError("save_preferences 至少提供一个有效偏好字段。")
    updated = {**preferences, **updates}
    if updated == preferences:
        return "偏好未变化，无需重复保存：" + json.dumps(preferences, ensure_ascii=False)
    preferences = updated
    content = json.dumps(preferences, ensure_ascii=False, indent=2)
    MEMORY_FILE.write_text(content, encoding="utf-8")
    return f"已保存用户偏好：{content}"


def weather_description(item: dict) -> str:
    for key in ("lang_zh", "weatherDesc"):
        descriptions = item.get(key)
        if isinstance(descriptions, list) and descriptions:
            first = descriptions[0]
            if isinstance(first, dict) and first.get("value"):
                return str(first["value"])
    return "天气描述未提供"


def get_weather(city: str, mode: str = "today", *, cache=None) -> str:
    if mode not in ("current", "today", "both"):
        raise ActionError("get_weather 的 mode 只能为 current、today 或 both。")
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
    url = f"https://wttr.in/{quote(city, safe='')}"
    key = (city, today)
    cached = cache.get(key) if cache is not None else None
    data = cached["data"] if cached is not None else None
    fetched_at = cached["fetched_at"] if cached is not None else None
    if data is None:
        response = requests.get(url, params={"format": "j1", "lang": "zh"}, timeout=20)
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError:
            raise ToolError("天气接口返回的内容不是有效 JSON。") from None
        fetched_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    if not isinstance(data, dict):
        raise ToolError("天气接口返回结构异常，应为 JSON 对象。")

    # 只缓存本轮成功使用的响应；失败结果不入缓存。
    views = ("current", "today") if mode == "both" else (mode,)
    parts, errors = [], []
    for view in views:
        try:
            parts.append(format_current(city, data) if view == "current" else format_forecast(city, data, today))
        except ToolError as error:
            errors.append(f"{view} 数据不可用：{error}")
    if errors:
        if cache is not None:
            cache.pop(key, None)  # 换 mode 后发现数据缺失，允许下一次重试重新联网。
        raise ToolError("\n".join(parts + errors) + f"\n来源：{url}")  # 保留另一部分已取得的信息。
    if cache is not None:
        cache[key] = {"data": data, "fetched_at": fetched_at}
    return "\n".join(parts) + f"\n数据获取时间：{fetched_at}；本次复用缓存：{cached is not None}\n来源：{url}"


def format_current(city: str, data: dict) -> str:
    conditions = data.get("current_condition")
    current = conditions[0] if isinstance(conditions, list) and conditions else {}
    if not isinstance(current, dict):
        raise ToolError("天气接口的 current_condition 结构异常。")
    if current.get("temp_C") is None:
        raise ToolError("天气接口未返回有效的当前气温。")
    observed = current.get("localObsDateTime")
    if observed:
        observed = f"{observed}（当地时间）"
    elif current.get("observation_time"):
        observed = f"{current['observation_time']}（接口原始时间，时区未标注）"
    else:
        observed = "未提供"
    return (
        f"{city}接口返回的当前观测：{weather_description(current)}，气温 {current['temp_C']}℃；"
        f"观测时间：{observed}。这是当前观测，不是全天预报；未确认观测新鲜度。"
    )


def format_forecast(city: str, data: dict, today: str) -> str:
    # 按用户所在的上海时区匹配今日日期，不将可能过期的观测日期当作今天。
    forecasts = data.get("weather")
    if not isinstance(forecasts, list) or any(not isinstance(x, dict) for x in forecasts):
        raise ToolError("天气接口的 weather 预报字段缺失或结构异常。")
    forecast = next((x for x in forecasts if x.get("date") == today), None)
    if forecast is None:
        raise ToolError(f"天气接口未返回 {today} 的预报，不能将其他日期称为今日天气。")
    hourly = forecast.get("hourly")
    if not isinstance(hourly, list) or not hourly:
        raise ToolError(f"天气接口未返回 {today} 的分时预报。")
    periods = []
    for item in hourly:
        if not isinstance(item, dict):
            raise ToolError("天气接口的 hourly 分时预报结构异常。")
        try:
            time_value = int(item["time"])
            if time_value < 0 or time_value > 2359 or time_value % 100 >= 60:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise ToolError("天气接口的分时预报时间缺失或无效。") from None
        details = [f"{time_value // 100:02d}:{time_value % 100:02d}", weather_description(item)]
        if item.get("tempC") is not None:
            details.append(f"{item['tempC']}℃")
        if item.get("chanceofrain") is not None:
            details.append(f"降雨概率 {item['chanceofrain']}%")
        if item.get("precipMM") is not None:
            details.append(f"预计降水 {item['precipMM']}mm")
        periods.append("，".join(details))
    low, high = forecast.get("mintempC"), forecast.get("maxtempC")
    temperatures = f"{low}～{high}℃" if low is not None and high is not None else "未提供"
    return (
        f"{city}日期 {today} 的全天预报（按用户上海时区匹配日期）：最低/最高温 {temperatures}。\n"
        "分时预报（接口返回时间）：\n" + "\n".join(periods)
        + "\n以上为预报，不是当前观测。"
    )


def get_attraction(city: str, requirements: str = "", query: str = "") -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ToolError("未配置 TAVILY_API_KEY，无法搜索景点。请补充配置。")
    preferences = load_preferences()
    search_query = query.strip() or " ".join(filter(None, [
        city, "景点 推荐", requirements or preferences.get("likes", "")
    ]))
    try:
        result = TavilyClient(api_key=api_key).search(
            query=search_query,
            search_depth="basic",
            include_answer=False,
            max_results=3,
        )
    except Exception as error:
        raise ToolError(f"景点搜索请求失败（{type(error).__name__}），请检查网络及搜索服务配置。") from None
    if not isinstance(result, dict) or not isinstance(result.get("results"), list):
        raise ToolError("搜索接口返回结构异常：results 缺失或不是列表。")
    results = result["results"]
    heading = f"搜索词：{search_query}\n结果数量：{len(results)}\n"
    if not results:
        return heading + "搜索正常完成，但未找到结果。可改写关键词重搜一次。"
    sources = []
    for index, item in enumerate(results, 1):
        if not isinstance(item, dict) or any(
            not isinstance(item.get(key), str) for key in ("title", "content", "url")
        ):
            raise ToolError("搜索接口结果的 title、content 或 url 字段缺失或类型错误。")
        if not item["title"].strip() or not item["url"].strip():
            raise ToolError("搜索接口结果缺少有效标题或来源链接。")
        content = item["content"].strip()
        if len(content) > 600:
            content = content[:600] + "（摘要已截断）"
        sources.append(
            f"{index}. {item['title']}\n摘要：{content or '未提供，无法据此核实推荐'}\n来源：{item['url']}"
        )
    return heading + "\n".join(sources)


TOOLS = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
    "save_preferences": save_preferences,
}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市天气。只问现在用 current，只需今日预报用 today；同时问当前天气并推荐景点时用 both，一次取得观测和预报。",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，例如杭州"},
                    "mode": {"type": "string", "enum": ["today", "current", "both"]},
                },
                "required": ["city", "mode"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_attraction",
            "description": "搜索指定城市的景点，返回网页摘要和来源；需结合用户限制及天气筛选。",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"},
                    "requirements": {"type": "string", "description": "本次要求或偏好，保留偏好与硬限制的区别；没有则传空字符串"},
                    "query": {"type": "string", "description": "包含城市及主题的简短搜索词；空字符串表示自动构造"},
                },
                "required": ["city", "requirements", "query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_preferences",
            "description": "保存用户明确表达的长期偏好；不保存本次临时限制。只更新有变化的字段，其余传 null。",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "likes": {"type": ["string", "null"], "description": "长期兴趣；不更新则传 null"},
                    "budget": {"type": ["string", "null"], "description": "常用预算；不更新则传 null"},
                    "transport": {"type": ["string", "null"], "description": "出行偏好；不更新则传 null"},
                },
                "required": ["likes", "budget", "transport"],
                "additionalProperties": False,
            },
        },
    },
]


def execute_tool(name: str, arguments: str, *, allowed_tools=None, weather_cache=None) -> str:
    """只执行白名单内的函数；JSON 参数仍需在本地校验。"""
    if name not in TOOLS:
        raise ActionError("工具不存在，仅允许已注册的工具。")
    if allowed_tools is not None and name not in allowed_tools:
        raise ActionError(f"当前子任务不允许调用 {name}，未执行工具。")
    try:
        kwargs = json.loads(arguments)
    except (TypeError, ValueError):
        raise ActionError("工具参数必须是有效的 JSON 对象。") from None
    if not isinstance(kwargs, dict):
        raise ActionError("工具参数必须是 JSON 对象。")
    tool = TOOLS[name]
    parameters = dict(inspect.signature(tool).parameters)
    if name == "get_weather":
        parameters.pop("cache", None)  # 缓存由程序传入，不允许模型提供。
    unknown = set(kwargs) - set(parameters)
    missing = [key for key, param in parameters.items()
               if param.default is inspect.Parameter.empty and key not in kwargs]
    problems = []
    if missing:
        problems.append("缺少必填参数：" + "、".join(missing))
    if unknown:
        problems.append("未知参数：" + "、".join(sorted(unknown)))
    if problems:
        raise ActionError(f"{name} " + "；".join(problems) + "。")
    for key, value in kwargs.items():
        if value is None and parameters[key].default is None:
            continue  # 偏好的 null 表示保留原值，不是删除已有偏好。
        if not isinstance(value, str):
            raise ActionError(f"{name} 的 {key} 必须为字符串。")
        kwargs[key] = value.strip()
        if not kwargs[key] and parameters[key].default != "":
            raise ActionError(f"{name} 的 {key} 必须为非空字符串。")
    if name == "get_weather":
        return tool(**kwargs, cache=weather_cache)
    return tool(**kwargs)
