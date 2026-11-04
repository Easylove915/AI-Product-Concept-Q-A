# -*- coding: utf-8 -*-
"""混合检索索引：BM25 + TF-IDF 余弦 +（可选）稠密向量，RRF 融合排序。

选这套组合的原因：
  * 关键词检索对「版本号、配置项、专有名词、错误码」这类精确查询更准；
  * 向量检索对「换个说法问同一件事」更稳；
  * RRF（倒数排名融合）不需要把不同量纲的分数归一化，工程上最省事也最稳。

索引本身持久化成一份人类可读的 JSON，重启服务无需重新拉取 GitHub。
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from config import Config
from rag.chunker import Chunk, Document, chunk_document
from rag.embeddings import cosine, get_backend
from rag.tokenizer import term_freq, tokenize

K1 = 1.5
B = 0.75

# RRF 融合权重。经典 RRF 三路等权，但这套组合里 BM25 与 TF-IDF 同属词袋模型、
# 排序高度相关，等于给「词面重叠」这个信号投了两票，语义召回被压成弱势方。
# 所以词袋两路各自降权，让「换个说法提问」时唯一能命中的向量那一路拿回应有分量。
# 具体数值由 eval 权重扫描确定，不是拍脑袋写的。
RRF_WEIGHTS = {"bm25": 1.0, "tfidf": 0.5, "dense": 2.0}


@dataclass
class Hit:
    chunk: Chunk
    rank: int
    rrf_score: float
    bm25: float
    tfidf: float
    dense: float
    relevance: float          # 0~100，仅用于前端展示的相对置信度

    def to_dict(self) -> dict:
        return {
            "chunk": self.chunk.to_dict(),
            "rank": self.rank,
            "rrf_score": round(self.rrf_score, 6),
            "bm25": round(self.bm25, 4),
            "tfidf": round(self.tfidf, 4),
            "dense": round(self.dense, 4),
            "relevance": round(self.relevance, 1),
        }


class HybridIndex:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.chunks: list[Chunk] = []
        self.documents: list[dict] = []
        self.meta: dict = {}
        # 以下为内存态，加载后由 _rebuild() 重建
        self._postings: dict[str, dict[int, float]] = {}
        self._idf: dict[str, float] = {}
        self._lengths: list[int] = []
        self._avgdl: float = 1.0
        self._tfidf_vecs: list[dict[str, float]] = []
        self._dense_vecs: list[list[float]] | None = None
        self._backend = None
        # 语料中从未出现过的词所用的 IDF（外域专有词的信息量上限）
        self._idf_unseen: float = 1.0
        # 最近一次检索的门控诊断信息，供上层生成「拒答」说明
        self.last_gate: dict = {}

    # ------------------------------------------------------------------ 构建
    def build(self, documents: list[Document], source: dict | None = None) -> None:
        self.chunks = []
        for doc in documents:
            self.chunks.extend(
                chunk_document(
                    doc,
                    chunk_size=self.cfg.chunk_size,
                    overlap=self.cfg.chunk_overlap,
                    min_chars=self.cfg.min_chunk_chars,
                )
            )
        self.documents = [
            {
                "doc_id": d.doc_id,
                "path": d.path,
                "title": d.title,
                "source": d.source,
                "github_url": d.github_url,
                "chars": len(d.text),
            }
            for d in documents
        ]
        self.meta = {
            "version": 1,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": source or {},
            "num_documents": len(documents),
            "num_chunks": len(self.chunks),
            "dense_mode": self.cfg.dense_mode,
            "chunk_size": self.cfg.chunk_size,
            "chunk_overlap": self.cfg.chunk_overlap,
        }
        self._dense_vecs = None
        self._backend = get_backend(self.cfg)
        if self._backend is not None:
            texts = [c.text for c in self.chunks]
            # 批大小由后端自己决定（百炼 text-embedding-v3 单次上限 10 条），
            # 索引层不关心具体是 10 还是 256。
            started = time.perf_counter()
            self._dense_vecs = self._backend.encode(texts)
            cost = time.perf_counter() - started
            self.meta["dense_model"] = getattr(self._backend, "model", None) or getattr(
                self._backend, "model_name", self._backend.name
            )
            self.meta["dense_dim"] = len(self._dense_vecs[0]) if self._dense_vecs else 0
            self.meta["dense_build_seconds"] = round(cost, 1)
            print(
                f"[index] 向量编码完成：{len(texts)} 片段 / "
                f"{self.meta['dense_dim']} 维 / {cost:.1f}s"
            )
        self._rebuild()

    def _rebuild(self) -> None:
        self._postings = {}
        self._lengths = []
        self._tfidf_vecs = []
        doc_freq: Counter = Counter()

        for idx, chunk in enumerate(self.chunks):
            tf = term_freq(chunk.text)
            self._lengths.append(sum(tf.values()))
            for term, freq in tf.items():
                self._postings.setdefault(term, {})[idx] = freq
                doc_freq[term] += 1

        n = len(self.chunks) or 1
        self._avgdl = (sum(self._lengths) / n) or 1.0
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }
        for tf in (term_freq(c.text) for c in self.chunks):
            vec = {
                term: (1 + math.log(freq)) * self._idf.get(term, 0.0)
                for term, freq in tf.items()
            }
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self._tfidf_vecs.append({k: v / norm for k, v in vec.items()})

        # 语料里完全没出现过的词（知识库外领域的专有词，如「报销」「限流」）
        # 按最高 IDF 处理：它们一旦缺失，就会显著拉低查询覆盖率，用于拒答判断。
        self._idf_unseen = math.log(1 + (n + 0.5) / 0.5)

    # ------------------------------------------------------------------ 持久化
    def save(self, path: Path | None = None) -> Path:
        target = path or self.cfg.index_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": self.meta,
            "documents": self.documents,
            "chunks": [c.to_dict() for c in self.chunks],
        }
        if self._dense_vecs is not None:
            # 向量降到 5 位小数再落盘：1024 维 × 489 片段的原始精度 JSON 约 10MB，
            # 降精度后体积减半，而余弦相似度在 1e-5 上的误差对排序没有影响。
            payload["dense"] = {
                "mode": self.cfg.dense_mode,
                "model": self.meta.get("dense_model"),
                "dim": self.meta.get("dense_dim"),
                "vectors": [[round(x, 5) for x in vec] for vec in self._dense_vecs],
            }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        return target

    def load(self, path: Path | None = None) -> bool:
        target = path or self.cfg.index_path
        if not target.exists():
            return False
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        self.meta = payload.get("meta", {})
        self.documents = payload.get("documents", [])
        self.chunks = [Chunk.from_dict(c) for c in payload.get("chunks", [])]
        dense = payload.get("dense")
        self._dense_vecs = dense["vectors"] if dense and dense.get("vectors") else None
        self._backend = get_backend(self.cfg) if self._dense_vecs is not None else None
        if self._backend is not None and self._dense_vecs:
            stored_dim = len(self._dense_vecs[0])
            if self._backend.dim != stored_dim:
                # 换了向量模型但没重建索引：维度不一致会让余弦被 zip() 静默截断，
                # 得到毫无意义的相似度。宁可关掉向量检索并提示重建。
                print(
                    f"[index] 索引里的向量是 {stored_dim} 维，当前模型是 "
                    f"{self._backend.dim} 维，已禁用稠密检索。请重新执行同步重建索引。"
                )
                self._dense_vecs = None
                self._backend = None
        self._rebuild()
        return True

    # ------------------------------------------------------------------ 检索
    def _bm25(self, query_tf: Counter) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for term in query_tf:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for idx, freq in self._postings.get(term, {}).items():
                dl = self._lengths[idx] or 1
                denom = freq + K1 * (1 - B + B * dl / self._avgdl)
                scores[idx] = scores.get(idx, 0.0) + idf * freq * (K1 + 1) / denom
        return sorted(scores.items(), key=lambda x: -x[1])

    def _tfidf(self, query_tf: Counter) -> list[tuple[int, float]]:
        qvec = {
            term: (1 + math.log(freq)) * self._idf.get(term, 0.0)
            for term, freq in query_tf.items()
        }
        norm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        qvec = {k: v / norm for k, v in qvec.items()}
        candidates: set[int] = set()
        for term in qvec:
            candidates.update(self._postings.get(term, {}).keys())
        scores = []
        for idx in candidates:
            dvec = self._tfidf_vecs[idx]
            # 只遍历较短的一侧
            small, large = (qvec, dvec) if len(qvec) < len(dvec) else (dvec, qvec)
            score = sum(v * large.get(k, 0.0) for k, v in small.items())
            if score > 0:
                scores.append((idx, score))
        return sorted(scores, key=lambda x: -x[1])

    def _dense(self, query: str) -> list[tuple[int, float]]:
        if self._backend is None or self._dense_vecs is None:
            return []
        qvec = self._backend.encode([query])[0]
        scores = [(i, cosine(qvec, vec)) for i, vec in enumerate(self._dense_vecs)]
        return sorted(scores, key=lambda x: -x[1])

    def _query_coverage(self, q_terms: set[str], idx: int) -> float:
        """IDF 加权的查询词覆盖率：这个片段「真正答到了」查询里的多少信息量。

        不用简单的命中比例，是因为「流程」「阶段」「怎么」这类高频词在语料里
        到处都是，命中它们几乎不说明相关性；而「RAG」「报销」这种低 DF 词
        才是判断相关性的关键。按 IDF 加权后，外域问题会因为关键专有词
        完全缺失而拿到很低的覆盖率，从而被门控拦下。
        """
        if not q_terms:
            return 0.0
        chunk_terms = self._tfidf_vecs[idx] if idx < len(self._tfidf_vecs) else {}
        total = 0.0
        hit = 0.0
        for term in q_terms:
            weight = self._idf.get(term, self._idf_unseen)
            # 中文单字（明 / 天 / 气 / 样）在中文语料里几乎处处命中，
            # 但它们几乎不携带领域信息，算进覆盖率会让「明天天气怎么样」
            # 这种外域问题拿到虚高分数。这里只降权，不改 BM25 的检索打分。
            if len(term) == 1 and "\u4e00" <= term <= "\u9fff":
                weight *= 0.4
            total += weight
            if term in chunk_terms:
                hit += weight
        return hit / total if total else 0.0

    def _rerank(self, query: str, rrf: dict[int, float], raw: dict | None = None) -> dict[int, float]:
        """轻量重排：用词面覆盖率、章节标题命中与语义相似度放大候选分数。

        纯 RRF 只保留名次，两篇文档分数 13.37 与 13.26 会被当成并列第一。
        这里用三个无模型成本的特征做二次打分，兼顾两类提问方式：
          * 词面特征（覆盖率 / 标题 / 拉丁锚点）——专有名词直问最准；
          * 语义特征（dense 余弦）——换个说法提问时词面必然对不上，
            只有向量能识别，不能让它被词面权重压回去。
        两个因子相乘：任一路给出强信号都能把候选顶上去，同时都不强时不会互相救场。
        """
        raw = raw or {}
        q_terms = set(tokenize(query))
        if not q_terms:
            return rrf
        strip = r"[\s，。？！、：:；;（）()「」【】\[\]·]+"
        q_compact = re.sub(strip, "", query).lower()
        # 拉丁字母/数字锚点：这批知识库的概念名多是 RAG / MCP / Badcase 这类术语，
        # 它们一旦命中基本就等于「问的就是这篇」，比中文二元组可靠得多。
        # 没有分词器时，这是唯一能低成本识别「专有名词」的信号。
        q_latin = {t for t in q_terms if t.isascii()}
        out: dict[int, float] = {}
        for idx, score in rrf.items():
            chunk = self.chunks[idx]
            coverage = self._query_coverage(q_terms, idx)
            heading_hits = sum(1 for term in q_terms if term in chunk.heading) / len(q_terms)
            latin_hit = 0.0
            if q_latin:
                chunk_latin = {t for t in self._tfidf_vecs[idx] if t.isascii()}
                latin_hit = len(q_latin & chunk_latin) / len(q_latin)
            # 概念名精确短语命中：wiki 类知识库里，用户常常直接拿页面标题提问
            title = re.sub(strip, "", chunk.doc_title).lower()
            title_hit = 0.0
            if len(title) >= 2 and (title in q_compact or q_compact in title):
                title_hit = 0.6 + 0.4 * min(1.0, len(title) / 8)
            lexical = 1 + 1.5 * coverage + 0.5 * heading_hits + 0.9 * latin_hit + title_hit

            semantic = 1.0
            if self._dense_vecs:
                # 实测这批语料上，无关文本的余弦也有 0.25~0.40（同属技术语域），
                # 所以从 0.45 起算增量，避免把「都挺像」的噪声也放大。
                sim = raw.get(idx, {}).get("dense", 0.0)
                semantic = 1.0 + 2.0 * max(0.0, sim - 0.45)
            # 两个因子相乘，不是取较大值。
            # 早先写的是 max(词面, 语义)，但词面因子上限约 4.9、语义上限约 2.4，
            # 词面永远赢 —— 结果就是「向量明明把正确文档排第一，又被重排按回第 4」。
            # 实测：max → Hit@1 70.8% / MRR 0.854，相乘 → 75.0% / 0.875，
            # 且 ss 取 2/3/4 结果完全一致，说明收益来自组合方式而非参数微调。
            out[idx] = score * lexical * semantic
        return out

    def search(self, query: str, top_k: int | None = None, gate: bool = True) -> list[Hit]:
        """gate=False 时跳过相关性门控，把判断权交给生成层。"""
        top_k = top_k or self.cfg.top_k
        if not self.chunks or not query.strip():
            return []
        query_tf = term_freq(query)
        if not query_tf:
            return []

        bm25 = self._bm25(query_tf)
        tfidf = self._tfidf(query_tf)
        dense = self._dense(query)
        limit = self.cfg.candidate_k

        rrf: dict[int, float] = {}
        raw: dict[int, dict[str, float]] = {}
        for name, ranked in (("bm25", bm25), ("tfidf", tfidf), ("dense", dense)):
            weight = RRF_WEIGHTS.get(name, 1.0)
            for rank, (idx, score) in enumerate(ranked[:limit], start=1):
                rrf[idx] = rrf.get(idx, 0.0) + weight / (self.cfg.rrf_k + rank)
                raw.setdefault(idx, {"bm25": 0.0, "tfidf": 0.0, "dense": 0.0})[name] = score

        if not rrf:
            # 查询词全部被停用词过滤掉时，用词面子串匹配兜底，避免直接空结果
            words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", query.lower())
            for idx, chunk in enumerate(self.chunks):
                text = chunk.text.lower()
                overlap = sum(1 for word in words if word in text)
                if overlap:
                    rrf[idx] = overlap / (self.cfg.rrf_k + 1)
            if not rrf:
                self.last_gate = {
                    "gated": True,
                    "coverage": 0.0,
                    "threshold": self.cfg.min_coverage,
                    "reason": "查询词完全没有出现在知识库中",
                }
                return []
        else:
            rrf = self._rerank(query, rrf, raw)
            # 相关性门控：知识库里根本没有这个话题时，宁可明确拒答，
            # 也不要硬凑 5 条片段让模型去编答案。
            best_idx = max(rrf, key=lambda i: rrf[i])
            coverage = self._query_coverage(set(query_tf), best_idx)
            blocked = (
                gate and self.cfg.min_coverage > 0 and coverage < self.cfg.min_coverage
            )
            self.last_gate = {
                "gated": blocked,
                "coverage": round(coverage, 4),
                "threshold": self.cfg.min_coverage,
            }
            if blocked:
                return []

        # 同一篇文档最多保留 3 个片段，避免答案被一篇长文档垄断
        per_doc: Counter = Counter()
        selected: list[tuple[int, float]] = []
        for idx, score in sorted(rrf.items(), key=lambda x: -x[1]):
            doc_id = self.chunks[idx].doc_id
            if per_doc[doc_id] >= 3:
                continue
            per_doc[doc_id] += 1
            selected.append((idx, score))
            if len(selected) >= top_k:
                break

        best = selected[0][1] or 1.0
        hits = []
        for rank, (idx, score) in enumerate(selected, start=1):
            values = raw.get(idx, {})
            hits.append(
                Hit(
                    chunk=self.chunks[idx],
                    rank=rank,
                    rrf_score=score,
                    bm25=values.get("bm25", 0.0),
                    tfidf=values.get("tfidf", 0.0),
                    dense=values.get("dense", 0.0),
                    relevance=round(100 * score / best, 1),
                )
            )

        if self.cfg.min_score > 0:
            hits = [h for h in hits if max(h.bm25, h.tfidf, h.dense) >= self.cfg.min_score]
        return hits

    # ------------------------------------------------------------------ 概览
    def summary(self) -> dict:
        return {
            "loaded": bool(self.chunks),
            "mode": self.cfg.mode_label,
            "dense_mode": self.cfg.dense_mode,
            **self.meta,
            "documents": self.documents,
            "index_path": str(self.cfg.index_path),
        }
