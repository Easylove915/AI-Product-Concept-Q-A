# 架构与实现细节

这份文档讲**为什么这么做**。只想跑起来的话看 [README](../README.md)。

## 分层

```
文档源 (GitHub / 本地目录)
   │  ingest.py      拉取、解析格式、统一成 Document
   ▼
Document ──chunker.py──▶ Chunk (带 doc_path / heading / line_start / line_end)
   │                      标题感知切节 + 重叠滑窗，保留行号用于溯源
   ▼
HybridIndex (index.py)
   ├─ 稀疏 1：BM25
   ├─ 稀疏 2：TF-IDF 余弦
   └─ 稠密 3：text-embedding-v3（可插拔，off / hashing / openai / local）
   │  RRF 融合 + 轻量重排，取 top-K
   ▼
generate.py   拼上下文（片段前标 1..n）→ LLM / 抽取式降级 → 引用校验
   ▼
Web / CLI     答案 + 引用卡片 + 检索明细表
```

| 层 | 文件 | 职责 | 生产环境可换成 |
| --- | --- | --- | --- |
| 接入 | `rag/ingest.py` | GitHub API 拉文档、解析格式 | Git clone / S3 / Confluence / 数据库 |
| 分块 | `rag/chunker.py` | 按标题切节 + 滑窗重叠 + 记录行号 | `RecursiveCharacterTextSplitter`、语义分块 |
| 分词 | `rag/tokenizer.py` | 中英文混合切词（中文二元组 + 停用词） | jieba / tiktoken / 模型 tokenizer |
| 索引检索 | `rag/index.py` | BM25 + TF-IDF + 向量，RRF 融合 + 重排 | Chroma / Qdrant / Milvus / Elasticsearch |
| 向量 | `rag/embeddings.py` | 可插拔 embedding 后端 | bge-m3 / text-embedding-3 / Cohere |
| 生成 | `rag/generate.py` | prompt、LLM 调用、引用校验、降级 | 加 cross-encoder 重排、多轮、流式 |
| 编排 | `rag/pipeline.py` | 串起来，对外三个动作 | FastAPI 依赖注入 / Celery 异步 |
| 服务/界面 | `app.py`、`web/` | HTTP API + 单页界面 | FastAPI + React/Vue |

**刻意的选择：检索层全部手写，不套 LangChain。**
理由不是「造轮子光荣」，而是这个项目唯一值得讲的东西就是 RRF 权重为什么是 1 / 0.5 / 2、
重排为什么不能取 `max`、门控阈值为什么会被中文二元组带偏。
换成框架后这些全变成一行 `Retriever(...)`，面试时打开文件指不出来。

---

## 为什么用混合检索

关键词检索对**版本号、错误码、配置项**这类精确查询更准；向量检索对「换个说法问同一件事」更稳。

RRF 按**排名**融合，不需要把两种量纲的分数归一化，工程上最省心 ——
但它也丢掉了分数幅度：BM25 13.37 和 13.26 会被当成并列第一。
所以后面补了一层**轻量重排**，用查询词覆盖率、章节标题命中和向量余弦做二次打分。

### 三路实测对比（24 道正题，top_k=5）

同一份索引、同一套题，只改融合权重：

| 检索方式 | Hit@1 | Recall@5 | MRR |
| --- | --- | --- | --- |
| 纯 BM25 | 66.7% | 100.0% | 0.810 |
| 纯 TF-IDF | 70.8% | 95.8% | 0.819 |
| 纯向量（text-embedding-v3，1024 维） | 70.8% | 100.0% | 0.837 |
| **混合（BM25 1.0 / TF-IDF 0.5 / 向量 2.0）** | **75.0%** | **100.0%** | **0.875** |

三个结论值得记：

1. **TF-IDF 会漏召**：Recall@5 只有 95.8%，纯向量补上来了 —— 这正是「换个说法提问」
   词面对不上时的典型表现。
2. **向量不是万能的**：纯向量在换说法题（B 类）上只有 3/8，比 BM25 的 4/8 还低。
   原因是这批概念名多是 RAG / MCP / Badcase 这类英文术语，词面命中就是最强信号。
3. **RRF 权重必须调，但不能瞎调**：BM25 与 TF-IDF 同属词袋模型、排序高度相关，
   三路等权等于给「词面重叠」投了两票，把向量压成弱势方。词袋两路降权（1.0 / 0.5）、
   向量加权到 2.0 后，Hit@1 从 66.7% 升到 70.8%。向量权重再加到 5.0 反而 MRR 掉到 0.844。

### 重排融合方式比权重更重要

第一版重排写的是 `max(词面分, 语义分)`，结果出了问题：

| 因子 | 量级上限 |
| --- | --- |
| 词面（覆盖率 × 标题命中） | ≈ 4.9 |
| 语义（向量余弦） | ≈ 2.4 |

`max` 实际上让词面**永远赢**，语义分只在线性无关时才起作用。表现是：
向量明明把正解排到了第一，重排又把它按回第 2–4 名
（A04：正解余弦 0.8457 vs 第一 0.8462；C01：正解 0.7230 vs 第一 0.6351，却排第 4）。

改成**两因子相乘**后 —— 只有两个信号同时高才算真命中：

| | `max` 融合 | 相乘融合 |
| --- | --- | --- |
| Hit@1 | 66.7% | **75.0%** |
| MRR | 0.833 | **0.875** |
| 跨文档题（C 类） | 5/6 | **6/6** |

而且缩放系数取 2 / 3 / 4 结果**完全一致** —— 说明收益来自组合方式本身，不是某个参数的运气。

---

## 接入真 embedding 的工程细节

百炼 `text-embedding-v3` 有三个必须处理的约束，都踩过：

| 约束 | 现象 | 处理 |
| --- | --- | --- |
| 单次 `input` 最多 **10 条** | 一次发 32 条直接 400 `batch size is invalid` | 批大小做成 `EMBEDDING_BATCH=10`，由后端内部切片，索引层不感知 |
| 单条请求约 **300ms** | 489 片段要 49 次请求，约 22s | 只在 `sync` 时编码一次，结果落盘复用 |
| 每次提问都要编码 query | 白等 300ms | query 向量做 LRU 缓存（512 条），重复提问零开销 |

另外两个坑：

- **维度必须校验**：换了向量模型但没重建索引时，余弦会被 `zip()` **静默截断**，
  得到一个毫无意义的相似度。`load()` 里比对存储维度，不一致就禁用向量检索并提示重建。
- **索引体积**：1024 维 × 489 片段，原始精度 JSON 约 10MB。向量降到 5 位小数后是 7.6MB，
  而余弦在 1e-5 上的误差对排序没有影响。

切换方式：`DENSE_MODE=off | hashing | openai | local`。
`hashing` 是不需要模型的伪向量，可以用来先验证「编码 → 余弦 → 融合」这条链路通不通；
`local` 需要 `pip install sentence-transformers`。

---

## 生成层与引用校验

生成时 system prompt 强制模型「只用给定片段回答、每个事实句末标 `[n]`、没信息就拒答」。
生成后后端还会校验编号是否指向真实片段，越界的引用会被剔除并给出警告。

**但要注意这个校验的边界**：`verify_citations()` 只校验「编号是否越界」，
它拦不住「编号合法但内容不符」的情况。这类失败的完整案例见
[failure-cases.md](failure-cases.md) 的 D03。

不配 Key 时走**抽取式降级**：从命中片段里挑最相关的句子，保留引用编号。
链路是完整的，只是不做归纳润色 —— 这是设计好的降级，不是 bug。

任意 OpenAI 兼容服务都能接：

```dotenv
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-xxxxxxxx
LLM_MODEL=deepseek-chat
```

通义、智谱、Moonshot、Ollama、vLLM 都兼容。本地 Ollama 把 `LLM_BASE_URL` 设成
`http://localhost:11434/v1`，Key 随便填个非空值。

---

## HTTP API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/status` | 索引状态、文档数、当前模式 |
| GET | `/api/docs` | 已索引文档列表 |
| POST | `/api/sync` | `{"repo","branch","path"}` → 重新同步 |
| POST | `/api/search` | `{"query","top_k"}` → 只返回检索命中 |
| POST | `/api/ask` | `{"query","top_k"}` → 完整答案 + 引用 + 检索明细 |

> `/api/search` 与 `/api/ask` 的**门控口径必须一致**（都用 `gate=not cfg.llm_enabled`）。
> 曾经 `search` 漏传这个参数，导致同一个问题 `ask` 能答、`search` 返回 0 命中。
