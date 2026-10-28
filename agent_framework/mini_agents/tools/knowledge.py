"""复用注册装饰器暴露只读检索；记忆维护和文件导入由应用明确执行。"""


def register_knowledge_tools(registry, *, memory=None, documents=None):
    if memory is not None:
        @registry.tool
        def search_memory(query: str) -> list[dict]:
            """检索当前用户明确保存的长期记忆。返回内容、记忆 ID；空数组表示未找到。"""
            return memory.search(query)

    if documents is not None:
        @registry.tool
        def search_documents(query: str) -> list[dict]:
            """检索当前用户文档，返回短预览、chunk_id 和 citation；需要正文时用 read_chunk。空数组表示未找到。"""
            results = []
            for item in documents.search(query):
                content = item["content"][:200]
                results.append({**item, "content": content, "offset": 0,
                                "next_offset": len(content) if len(content) < item["total_chars"] else None,
                                "truncated": len(content) < item["total_chars"]})
            return results

        @registry.tool
        def read_chunk(chunk_id: str, offset: int = 0, max_chars: int = 1000) -> dict:
            """按 chunk_id 读取已索引原文。offset 为字符偏移；按返回的 next_offset 继续读取，null 表示已读完。回答引用 citation。"""
            return documents.read_chunk(chunk_id, offset=offset, max_chars=max_chars)
    return registry
