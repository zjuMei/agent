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
            """检索当前用户的本地文档。返回原文片段和 citation，回答时引用 citation；空数组表示未找到。"""
            return documents.search(query)
    return registry
