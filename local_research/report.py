"""检查引用的存在性和读取记录；语义是否被原文支持仍需核对。"""

import json
from pathlib import Path
import re


SECTIONS = ("结论摘要", "分问题分析", "对比与适用条件", "资料不足与未解决问题")


def validate_report(text, evidence):
    errors = []
    if not text.strip():
        errors.append("报告正文为空")
    if not re.search(r"^#\s+\S", text, re.M):
        errors.append("缺少报告标题")
    for section in SECTIONS:
        if not re.search(rf"^##\s+{re.escape(section)}\s*$", text, re.M):
            errors.append(f"缺少章节：{section}")
    if re.search(r"^#{1,6}\s*(参考资料|参考文献|References)\s*$", text, re.M | re.I):
        errors.append("来源列表由程序生成，请勿在报告正文中自行添加")
    available = {item["citation"]: item for item in evidence}
    used = []
    # 去掉代码块和行内代码，示例中的编号不算对报告的引用。
    prose = re.sub(r"```.*?```|~~~.*?~~~|`[^`\n]*`", "", text, flags=re.S)
    for match in re.finditer(r"(?<!\\)\[([sS][^\]\n]*)\]", prose):
        citation = match.group(1)
        if not re.fullmatch(r"S[1-9][0-9]*", citation):
            errors.append(f"引用格式无效：[{citation}]，应使用独立的 [S1] 格式")
        elif citation not in available:
            errors.append(f"引用不存在：[{citation}]")
        elif not available[citation]["pages"]:
            errors.append(f"引用尚未读取正文：[{citation}]")
        elif citation not in used:
            used.append(citation)
        if prose[match.end():].startswith(("(", "[", ":")) and not prose[match.end():].startswith("[S"):
            errors.append(f"引用应为普通编号，不能附加链接或定义：[{citation}]")
    if not used:
        errors.append("报告未引用任何已读取的证据")
    return list(dict.fromkeys(errors)), used


def append_sources(text, evidence, used):
    items = {item["citation"]: item for item in evidence}
    references = ["## 参考资料", ""]
    for citation in used:
        item = items[citation]
        references.append(f"- [{citation}] `{item['source']}`，L{item['start_line']}–L{item['end_line']}")
    return text.rstrip() + "\n\n" + "\n".join(references) + "\n"


def insufficient_report(question):
    return (f"# 本地资料研究结果\n\n研究问题：{question}\n\n"
            "## 结论摘要\n\n资料不足：本次未成功读取可引用的非空原文，无法据此形成研究结论。\n\n"
            "## 资料不足与未解决问题\n\n请检查资料内容、检索关键词或补充相关 Markdown/TXT 文件后重新研究。\n")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")
