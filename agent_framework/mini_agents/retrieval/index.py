"""本地小型知识库：应用导入文件，模型只能检索片段。"""

import hashlib
from pathlib import Path

from ..core.retrieval import bounded, rank
from .documents import split_document


class DocumentIndex:
    def __init__(self, db, *, namespace="local"):
        if not namespace.strip():
            raise ValueError("namespace 不能为空")
        self.db, self.namespace = db, namespace

    def add_file(self, path, *, chunk_size=700, overlap=80):
        path = Path(path).expanduser().resolve()
        if path.suffix.lower() not in (".md", ".txt"):
            raise ValueError("基础知识库只支持 UTF-8 编码的 .md 和 .txt 文件")
        text = path.read_text(encoding="utf-8-sig")
        source = str(path)
        chunks = split_document(text, chunk_size=chunk_size, overlap=overlap)
        with self.db.transaction() as connection:
            # 同路径重新导入时原子替换旧片段；空文件也会清除过时内容。
            connection.execute("DELETE FROM chunks WHERE namespace=? AND source=?",
                               (self.namespace, source))
            for index, chunk in enumerate(chunks):
                digest = hashlib.sha256(f"{source}\0{index}\0{chunk['content']}".encode()).hexdigest()[:20]
                connection.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                    (self.namespace, source, digest, chunk["content"],
                     chunk["start_line"], chunk["end_line"]),
                )
        return {"source": source, "chunks": len(chunks)}

    def add_directory(self, path):
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"知识库目录不存在：{root}")
        results = []
        for file in sorted(root.rglob("*")):
            if (file.is_file() and file.suffix.lower() in (".md", ".txt")
                    and file.resolve().is_relative_to(root)):
                results.append(self.add_file(file))
        return results

    def search(self, query, *, limit=3, max_chars=3000):
        with self.db.transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM chunks WHERE namespace=? ORDER BY source, start_line, chunk_id",
                (self.namespace,),
            ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["citation"] = f"{Path(item['source']).name}:L{item['start_line']}-L{item['end_line']}"
            item["total_chars"] = len(item["content"])
            items.append(item)
        return bounded(rank(query, items), limit=limit, max_chars=max_chars)

    def read_chunk(self, chunk_id, *, offset=0, max_chars=1000):
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("chunk_id 不能为空")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset 必须是非负整数")
        if type(max_chars) is not int or not 1 <= max_chars <= 2000:
            raise ValueError("max_chars 必须是 1 至 2000 的整数")
        with self.db.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM chunks WHERE namespace=? AND chunk_id=?", (self.namespace, chunk_id),
            ).fetchone()
        if row is None:
            raise ValueError("未找到当前用户的片段，请重新搜索；文档重导入后旧片段可能失效")
        item = dict(row)
        total = len(item["content"])
        if offset > total:
            raise ValueError("offset 超过片段长度")
        end = min(total, offset + max_chars)
        item.update(content=item["content"][offset:end], total_chars=total, offset=offset,
                    next_offset=end if end < total else None, truncated=offset > 0 or end < total)
        # 引用范围属于整个原始片段；页内位置由 offset 表示，不读取磁盘上的其他文件。
        item["citation"] = f"{Path(item['source']).name}:L{item['start_line']}-L{item['end_line']}"
        return item

    def remove(self, path):
        with self.db.transaction() as connection:
            return connection.execute(
                "DELETE FROM chunks WHERE namespace=? AND source=?",
                (self.namespace, str(Path(path).expanduser().resolve())),
            ).rowcount
