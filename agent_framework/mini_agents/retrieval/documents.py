"""按段落/行优先切分文本，长段落使用带重叠的窗口，保留原始行号。"""


def split_document(text, *, chunk_size=700, overlap=80):
    if (type(chunk_size) is not int or type(overlap) is not int
            or not 0 <= overlap < chunk_size):
        raise ValueError("需要 0 <= overlap < chunk_size，且均为整数")
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for separator in ("\n\n", "\n"):
                boundary = text.rfind(separator, start + chunk_size // 2, end)
                if boundary != -1:
                    end = boundary + len(separator)
                    break
        raw = text[start:end]
        content = raw.strip()
        if content:
            left = start + len(raw) - len(raw.lstrip())
            right = end - (len(raw) - len(raw.rstrip()))
            chunks.append({"content": content, "start_line": text.count("\n", 0, left) + 1,
                           "end_line": text.count("\n", 0, right - 1) + 1})
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks
