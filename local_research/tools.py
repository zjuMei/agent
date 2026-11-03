"""两个只读研究工具：复用索引，并保存本次实际返回的证据。"""

from copy import deepcopy
from pathlib import Path

from agent_framework.mini_agents import ToolRegistry


class ResearchTools:
    def __init__(self, documents, root):
        self.documents = documents
        self.root = Path(root).resolve()
        self._evidence = {}  # chunk_id -> 本次任务内稳定的 S 编号及读取记录
        self.registry = ToolRegistry()
        self.registry.tool(self.search_documents)
        self.registry.tool(self.read_chunk)

    def _identify(self, item):
        chunk_id = item["chunk_id"]
        if chunk_id not in self._evidence:
            self._evidence[chunk_id] = {
                "citation": f"S{len(self._evidence) + 1}",
                "chunk_id": chunk_id,
                "source": Path(item["source"]).relative_to(self.root).as_posix(),
                "start_line": item["start_line"], "end_line": item["end_line"],
                "total_chars": item["total_chars"],
                "preview": item["content"][:200], "pages": [],
            }
        evidence = self._evidence[chunk_id]
        return {**item, "citation": evidence["citation"], "source": evidence["source"]}

    def search_documents(self, query: str) -> list[dict]:
        """搜索本次资料，返回最多三个短预览、chunk_id、citation。引用前用 read_chunk 阅读正文。"""
        results = []
        for item in self.documents.search(query):
            item = self._identify(item)
            content = item["content"][:200]
            results.append({**item, "content": content, "offset": 0,
                            "next_offset": len(content) if len(content) < item["total_chars"] else None,
                            "truncated": len(content) < item["total_chars"]})
        return results

    def read_chunk(self, chunk_id: str, offset: int = 0, max_chars: int = 1000) -> dict:
        """读取搜索返回的 chunk_id 原文。引用 citation；需要后续内容时使用实际 next_offset。"""
        if chunk_id not in self._evidence:
            raise ValueError("请先 search_documents，再使用搜索返回的 chunk_id")
        item = self._identify(self.documents.read_chunk(chunk_id, offset=offset, max_chars=max_chars))
        page = {key: item[key] for key in ("offset", "next_offset", "content")}
        pages = self._evidence[chunk_id]["pages"]
        if item["content"] and page not in pages:
            pages.append(page)
        return item

    def snapshot(self):
        return deepcopy(list(self._evidence.values()))
