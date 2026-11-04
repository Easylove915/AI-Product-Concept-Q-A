# -*- coding: utf-8 -*-
"""稠密向量后端（可插拔）。

Demo 默认关闭稠密检索（DENSE_MODE=off），只用稀疏检索即可完整演示 RAG。
这个模块演示的是"以后怎么把向量模型接进来"：只要实现 encode(texts)，
索引层不用改一行代码。

  off      : 不使用（默认，零依赖）
  hashing  : 内置 hash trick 伪向量，用于演示向量检索链路，无需模型
  openai   : OpenAI 兼容 /embeddings 接口（含百炼 DashScope、SiliconFlow 等）
  local    : sentence-transformers 本地模型（需 pip install sentence-transformers）

接入 text-embedding-v3 的实测结论（写进 README，面试可讲）：
  * 维度 1024，单次 input 上限 10 条，超过返回 400 InvalidParameter；
  * 单条请求约 300ms，489 个片段需要 49 次请求（约 17s）；
  * 因此批大小必须可配，且检索时的 query 向量必须缓存，否则每次提问
    都要多花 300ms 在「把同一个问题再编码一遍」上。
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from collections import OrderedDict

from config import Config
from rag.tokenizer import tokenize


def _l2(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class _CachedBackend:
    """给 encode 加一层「文本 → 向量」缓存。

    检索时每次提问都会编码一次 query，同一个问题重复问、或者评测时反复
    跑同一批用例，命中缓存能省掉一次网络往返（实测 ~300ms）。
    只缓存短文本（query），批量编码语料时直接跳过，不占内存。
    """

    cache_size = 512

    def __init__(self) -> None:
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _cache_get(self, text: str):
        vec = self._cache.get(text)
        if vec is None:
            self.misses += 1
            return None
        self._cache.move_to_end(text)
        self.hits += 1
        return vec

    def _cache_put(self, text: str, vec: list[float]) -> None:
        self._cache[text] = vec
        self._cache.move_to_end(text)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)


class HashingBackend(_CachedBackend):
    """把 token 哈希到固定维度桶里，得到一个确定性的稠密向量。

    它不是语义向量，但足以验证「向量编码 → 余弦召回 → 融合排序」这条链路。
    """
    name = "hashing"
    dim = 256
    batch_size = 256

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            cached = self._cache_get(text)
            if cached is not None:
                vectors.append(cached)
                continue
            vec = [0.0] * self.dim
            for token in tokenize(text):
                bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16) % self.dim
                vec[bucket] += 1.0
            vec = _l2(vec)
            self._cache_put(text, vec)
            vectors.append(vec)
        return vectors


class OpenAIBackend(_CachedBackend):
    """OpenAI 兼容的 /embeddings 接口。

    适配百炼（DashScope compatible-mode）、SiliconFlow、OpenAI 等，
    复用 LLM_BASE_URL / LLM_API_KEY，不需要第二套凭证。
    """

    name = "openai"
    # 429 / 5xx 重试，最多 3 次，退避 1s / 2s / 4s
    max_retries = 3

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.model = cfg.embedding_model
        # 先按配置假设维度，第一次真实响应回来后以接口返回为准
        self.dim = cfg.embedding_dim or 1024
        self.batch_size = max(1, cfg.embedding_batch)

    def _post(self, batch: list[str]) -> list[list[float]]:
        url = self.cfg.llm_base_url.rstrip("/") + "/embeddings"
        body = json.dumps({"model": self.model, "input": batch}).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.llm_api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=self.cfg.llm_timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        items = sorted(payload["data"], key=lambda item: item["index"])
        vectors = [_l2(item["embedding"]) for item in items]
        # 以接口返回的维度为准，避免配置写错导致余弦计算静默截断
        if vectors:
            self.dim = len(vectors[0])
        return vectors

    def _post_with_retry(self, batch: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._post(batch)
            except urllib.error.HTTPError as exc:
                # 4xx 里只有 429 值得重试，其余（参数错、余额不足）重试也没用
                if exc.code != 429 and exc.code < 500:
                    detail = exc.read().decode("utf-8", "ignore")[:300]
                    raise RuntimeError(f"向量接口返回 {exc.code}：{detail}") from exc
                last_error = exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
            if attempt < self.max_retries - 1:
                time.sleep(2**attempt)
        raise RuntimeError(f"向量接口连续 {self.max_retries} 次失败：{last_error}")

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float] | None] = [self._cache_get(t) for t in texts]
        pending = [i for i, v in enumerate(vectors) if v is None]
        if pending:
            unique = list(dict.fromkeys(texts[i] for i in pending))
            encoded: dict[str, list[float]] = {}
            for start in range(0, len(unique), self.batch_size):
                batch = unique[start : start + self.batch_size]
                for text, vec in zip(batch, self._post_with_retry(batch)):
                    encoded[text] = vec
            for i in pending:
                vectors[i] = encoded[texts[i]]
            # 只缓存短文本（query），避免把整份语料都塞进内存缓存
            for text, vec in encoded.items():
                if len(text) <= self.cache_size_chars:
                    self._cache_put(text, vec)
        return [v for v in vectors if v is not None]

    cache_size_chars = 200


class LocalBackend(_CachedBackend):
    name = "local"
    batch_size = 32

    def __init__(self, cfg: Config | None = None):
        super().__init__()
        self.model_name = (cfg.embedding_model if cfg else "") or (
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        from sentence_transformers import SentenceTransformer  # 延迟导入
        self.model = SentenceTransformer(self.model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = [self._cache_get(t) for t in texts]
        pending = [i for i, v in enumerate(vectors) if v is None]
        if pending:
            batch = [texts[i] for i in pending]
            encoded = self.model.encode(batch, batch_size=self.batch_size)
            for i, vec in zip(pending, encoded):
                vectors[i] = _l2(vec.tolist())
            for i in pending:
                self._cache_put(texts[i], vectors[i])
        return [v for v in vectors if v is not None]


def get_backend(cfg: Config):
    """返回 backend 实例；不可用则返回 None（调用方自动降级）。"""
    mode = cfg.dense_mode
    if mode == "off":
        return None
    try:
        if mode == "hashing":
            return HashingBackend()
        if mode == "openai":
            if not cfg.llm_api_key:
                raise RuntimeError("需要 LLM_API_KEY")
            return OpenAIBackend(cfg)
        if mode == "local":
            return LocalBackend(cfg)
        raise RuntimeError(f"未知 DENSE_MODE={mode}")
    except Exception as exc:
        print(f"[embeddings] 稠密向量后端不可用，已降级为稀疏检索：{exc}")
        return None
