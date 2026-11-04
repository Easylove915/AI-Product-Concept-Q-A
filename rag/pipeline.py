# -*- coding: utf-8 -*-
"""编排层：把 ingest → chunk → index → retrieve → generate 串成一条链路。

对外只暴露三个动作：status / sync / ask，Web 层与 CLI 层共用同一个服务实例。
"""
from __future__ import annotations

import threading

from config import CFG, Config
from rag.generate import answer
from rag.index import HybridIndex
from rag.ingest import load_documents


class RAGService:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.index = HybridIndex(cfg)
        self.lock = threading.Lock()
        self.last_error = ""
        # 启动即有索引：优先读磁盘缓存，否则用内置示例文档自动构建
        if not self.index.load():
            try:
                self.sync()
            except Exception as exc:
                self.last_error = f"自动构建示例索引失败：{exc}"

    def status(self) -> dict:
        return {
            **self.index.summary(),
            "github_repo": self.cfg.github_repo,
            "github_path": self.cfg.github_path,
            "github_branch": self.cfg.github_branch,
            "docs_dir": str(self.cfg.docs_dir),
            "top_k": self.cfg.top_k,
            "last_error": self.last_error,
        }

    def sync(self, overrides: dict | None = None) -> dict:
        """重新拉取文档并重建索引。overrides 允许前端临时指定仓库参数。"""
        overrides = overrides or {}
        with self.lock:
            cfg = self.cfg
            # 临时覆盖（不写回 .env），用于界面上试不同仓库
            backup = {}
            for key, value in overrides.items():
                if value and hasattr(cfg, key):
                    backup[key] = getattr(cfg, key)
                    setattr(cfg, key, value)
            try:
                documents = load_documents(cfg)
                if not documents:
                    raise RuntimeError("没有读取到任何非空文档")
                source = {
                    "type": "github" if cfg.github_repo else "local",
                    "repo": cfg.github_repo,
                    "branch": cfg.github_branch,
                    "path": cfg.github_path,
                    "docs_dir": str(cfg.docs_dir) if not cfg.github_repo else "",
                }
                self.index.build(documents, source)
                path = self.index.save()
                self.last_error = ""
                return {
                    "ok": True,
                    "num_documents": len(documents),
                    "num_chunks": len(self.index.chunks),
                    "source": source,
                    "index_path": str(path),
                    "documents": self.index.documents,
                }
            except Exception as exc:
                self.last_error = str(exc)
                raise
            finally:
                for key, value in backup.items():
                    setattr(cfg, key, value)

    def ask(self, query: str, top_k: int | None = None) -> dict:
        query = (query or "").strip()
        if not query:
            return {"error": "问题不能为空"}
        with self.lock:
            # 拒答分两层，按能力分工：
            #   配了 LLM → 关掉规则门控，把「答不答」交给模型判断。
            #     模型看到片段都在讲别的主题时会自己拒答，比词面覆盖率准得多，
            #     也不会因为中文跨词边界的伪二元组误杀正常问题。
            #   没配 LLM → 只能用规则门控兜底，至少拦掉明显外域的问题。
            hits = self.index.search(
                query, top_k=top_k, gate=not self.cfg.llm_enabled
            )
            return answer(query, hits, self.cfg, gate=self.index.last_gate)

    def search(self, query: str, top_k: int | None = None) -> dict:
        query = (query or "").strip()
        if not query:
            return {"error": "查询不能为空"}
        with self.lock:
            # 门控口径必须和 ask() 一致：配了 LLM 就不该用规则拦，
            # 否则会出现「ask 能答、search 却 0 命中」的自相矛盾。
            hits = self.index.search(
                query, top_k=top_k, gate=not self.cfg.llm_enabled
            )
            return {"query": query, "hits": [h.to_dict() for h in hits]}


SERVICE = RAGService()
