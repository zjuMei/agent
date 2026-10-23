"""基础词面匹配：英文按词、中文按相邻双字切分，不调用模型。"""

import math
import re
from collections import Counter


def terms(text):
    tokens = []
    for part in re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]+", text.lower()):
        if "\u3400" <= part[0] <= "\u9fff":
            tokens.extend(part[i:i + 2] for i in range(len(part) - 1))
            if len(part) == 1:
                tokens.append(part)
        else:
            tokens.append(part)
    return Counter(tokens)


def rank(query, items):
    """items 中的 content 为待检索文本；只返回有词面交集的条目。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("检索问题不能为空")
    query_terms = terms(query)
    query_norm = math.sqrt(sum(value * value for value in query_terms.values()))
    if not query_norm:
        return []
    results = []
    for item in items:
        counts = terms(item["content"])
        overlap = sum(value * counts[token] for token, value in query_terms.items())
        if overlap:
            norm = math.sqrt(sum(value * value for value in counts.values()))
            results.append({**item, "score": round(overlap / (query_norm * norm), 6)})
    return sorted(results, key=lambda item: item["score"], reverse=True)


def bounded(items, *, limit, max_chars):
    """限制条数和正文总字符数；元数据保留，截断时明确标记。"""
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("limit 必须是 1 至 20 的整数")
    if type(max_chars) is not int or not 1 <= max_chars <= 10000:
        raise ValueError("max_chars 必须是 1 至 10000 的整数")
    results = []
    for item in items[:limit]:
        if max_chars <= 0:
            break
        content = item["content"][:max_chars]
        results.append({**item, "content": content,
                        "truncated": item.get("truncated", False) or len(content) < len(item["content"])})
        max_chars -= len(content)
    return results
