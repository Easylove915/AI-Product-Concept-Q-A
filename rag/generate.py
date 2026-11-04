# -*- coding: utf-8 -*-
"""答案生成与引用溯源。

两条路径：
  1) 配置 LLM_API_KEY → 调用 OpenAI 兼容 /chat/completions，prompt 强制 [n] 引用；
  2) 未配置 / 调用失败 → 抽取式降级，从命中片段里挑最相关的句子并保留引用编号。
     Demo 因此永远可以演示完整链路，不会因为缺 Key 卡住。

生成之后统一做「引用校验」：答案里出现的每个 [n] 都必须能映射到真实 chunk，
否则给出警告。这是 RAG 溯源可信度的第一道闸门。
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

from config import Config
from rag.index import Hit
from rag.tokenizer import tokenize

SYSTEM_PROMPT = """你是一个企业知识库问答助手。严格遵守以下规则：

1. 只使用下面提供的「知识库片段」回答，不要使用你的先验知识补充或推断。
2. 每一个事实性句子末尾必须用 [编号] 标注来源，例如：配额是每分钟 100 次 [2]。可以引用多个片段，如 [1][3]。
3. 编号必须对应片段前的 [n] 标记，不得编造不存在的编号，不得改动片段原文的含义。
4. 如果片段中没有足够信息，直接回答「知识库中没有找到相关说明」，并说明还缺少什么信息，不要猜测。
5. 用简洁的中文回答，先给结论，再给必要的条件/步骤。不要复述问题，不要输出「根据资料」这类套话。
6. 片段里用作讲解的**示例、演示、模拟输出、测试数据都是虚构的**（例如文档为了讲接口调用，虚构了一段「某城市明日晴、气温 25 度」的返回值）。这类内容不是知识库陈述的事实，不得当作答案，也不得替换其中的主体后复用。
7. 如果用户问的主体或对象在片段中根本没有出现，不要用片段里另一个主体的数据代答。例如片段只讲了 A 城市的情况，而用户问 B 城市，必须说明「知识库中没有关于 B 城市的说明」。
8. 如果用户问的是实时信息（天气、股价、汇率、新闻、当前时间等），直接说明知识库无法提供这类数据。

知识库片段：
---
{context}
---
"""

_CITE_RE = re.compile(r"\[(\d{1,2})\]")
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*|\n+")
# 模型按 prompt 第 4 条拒答时的措辞。拒答本来就不带引用，
# 不能因此报「未标注引用」的警告，否则界面上每条拒答都挂个黄条。
_NO_INFO_RE = re.compile(r"知识库中(没有|未)|没有(找到|收录|相关)|未(找到|收录)|不在.{0,12}范围")


def build_context(hits: list[Hit], max_chars: int = 6000) -> str:
    blocks = []
    used = 0
    for hit in hits:
        chunk = hit.chunk
        header = f"[{hit.rank}] 来源：{chunk.doc_path} › {chunk.heading} (L{chunk.line_start}-L{chunk.line_end})"
        block = f"{header}\n{chunk.content}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def _call_llm(query: str, context: str, cfg: Config) -> tuple[str, str]:
    url = cfg.llm_base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg.llm_model,
        "temperature": cfg.llm_temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": query},
        ],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {cfg.llm_api_key}",
        },
    )
    with urllib.request.urlopen(request, timeout=cfg.llm_timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"].strip(), cfg.llm_model


def _sentence_score(sentence: str, q_terms: set[str]) -> float:
    """给句子打「信息量」分，避免只挑中引导句或空泛句。"""
    s_terms = set(tokenize(sentence))
    score = 2.0 * len(q_terms & s_terms) / (len(q_terms) or 1)
    if 20 <= len(sentence) <= 160:
        score += 0.15
    if len(sentence) < 16:
        score -= 0.35
    # 含数字、代码标识符、英文术语，通常是真正承载答案的句子
    if re.search(r"\d|`[^`]+`|[A-Za-z]{3,}", sentence):
        score += 0.18
    if sentence.endswith(("：", ":", "，")):
        score -= 0.25
    return score


def _clean_sentences(content: str) -> list[tuple[str, bool]]:
    """切句并把 Markdown 表格行转成可读短句，返回 (句子, 是否为表格行)。

    这批 wiki 大量用表格承载核心信息，直接按行摘录会得到一堆裸管道符；
    转成「A · B · C」后既可读，又能在打分时降权，优先让正文句子进答案。
    """
    out: list[tuple[str, bool]] = []
    for raw in _SENT_SPLIT.split(content):
        sentence = raw.strip()
        if len(sentence) < 10:
            continue
        if sentence.lstrip().startswith("|"):
            cells = [
                re.sub(r"\*+", "", c).strip("` ").strip()
                for c in sentence.strip().strip("|").split("|")
            ]
            cells = [c for c in cells if c and not re.fullmatch(r"[-:\s]+", c)]
            if not cells:
                continue
            # 纯表头行（全是短标签、无数字与问号）对答案没有信息量
            if all(len(c) <= 8 for c in cells) and not any(
                re.search(r"\d|[？?。]", c) for c in cells
            ):
                continue
            text = " · ".join(cells)
            if len(text) < 12:
                continue
            out.append((text, True))
        else:
            sentence = re.sub(r"^\s*[-•*]\s+", "", sentence).replace("**", "").strip()
            if len(sentence) >= 10:
                out.append((sentence, False))
    return out


def _extractive(query: str, hits: list[Hit]) -> str:
    """无 LLM 时的降级：每个命中片段挑信息量最高的原句，按检索排名组织。

    按片段分组而不是全局选句，能保证答案覆盖多个来源，也不会被某一篇
    长文档里恰好含查询词的短句带偏。
    """
    q_terms = set(tokenize(query))
    picked: list[tuple[int, str]] = []
    seen: set[str] = set()

    for hit in hits[:2]:
        pairs = _clean_sentences(hit.chunk.content)
        if not pairs:
            continue
        sentences = [p[0] for p in pairs]
        is_table = [p[1] for p in pairs]

        def overlap(i: int) -> float:
            return len(q_terms & set(tokenize(sentences[i]))) / (len(q_terms) or 1)

        def score(i: int) -> float:
            return _sentence_score(sentences[i], q_terms) - (0.25 if is_table[i] else 0.0)

        # 第二名片段只收正文句子且要求更高的词面重叠，防止答案跑题
        ranked = sorted(
            (
                i for i in range(len(sentences))
                if hit.rank == 1 or (overlap(i) >= 0.28 and not is_table[i])
            ),
            key=lambda i: (-overlap(i), -score(i)),
        )
        take = 4 if hit.rank == 1 else 1
        chosen: list[tuple[int, str]] = []
        consumed: set[int] = set()
        for i in ranked:
            if i in consumed:
                continue
            sentence = sentences[i]
            # 引导句（「按以下顺序排查：」）后面紧跟的才是正文，一并带上
            if sentence.endswith(("：", ":")) and i + 1 < len(sentences):
                sentence = sentence + " " + sentences[i + 1]
                consumed.add(i + 1)
            key = sentence[:24]
            if key in seen:
                continue
            seen.add(key)
            chosen.append((i, sentence))
            if len(chosen) >= take:
                break
        for _, sentence in sorted(chosen, key=lambda item: item[0]):
            picked.append((hit.rank, sentence))

    if not picked:
        return "知识库中没有找到与该问题直接相关的说明。"

    lines = [f"· {sentence} [{rank}]" for rank, sentence in picked[:5]]
    return (
        "【检索式答案】当前未配置大模型，以下直接摘录知识库中最相关的原句；"
        "配置 LLM_API_KEY 后可获得归纳润色后的生成式答案。\n\n" + "\n\n".join(lines)
    )


def verify_citations(answer: str, hits: list[Hit]) -> tuple[list[dict], list[str]]:
    """把答案中的 [n] 映射到真实 chunk，并返回告警。"""
    warnings: list[str] = []
    indexes: list[int] = []
    for raw in _CITE_RE.findall(answer):
        num = int(raw)
        if num not in indexes:
            indexes.append(num)
    valid = [n for n in indexes if 1 <= n <= len(hits)]
    invalid = [n for n in indexes if n < 1 or n > len(hits)]
    if invalid:
        warnings.append(f"答案引用了不存在的片段编号 {invalid}，已忽略。")
    if not indexes:
        # 模型主动拒答时不提示引用缺失——拒答本来就不该有引用
        if not _NO_INFO_RE.search(answer):
            warnings.append("答案未标注任何引用编号，建议人工核对。")

    citations = [
        {
            "index": n,
            **hits[n - 1].chunk.to_dict(),
            "relevance": hits[n - 1].relevance,
            "bm25": round(hits[n - 1].bm25, 3),
            "tfidf": round(hits[n - 1].tfidf, 3),
            "dense": round(hits[n - 1].dense, 3),
        }
        for n in valid
    ]
    return citations, warnings


def answer(query: str, hits: list[Hit], cfg: Config, gate: dict | None = None) -> dict:
    started = time.time()
    if not hits:
        text = "知识库中没有检索到相关内容，无法回答。请先同步文档，或换一种提问方式。"
        warnings = ["检索无命中"]
        if gate and gate.get("gated"):
            coverage = gate.get("coverage", 0.0)
            threshold = gate.get("threshold", 0.0)
            text = (
                f"知识库中没有与该问题相关的内容。\n\n"
                f"判定依据：这个问题里最关键的词在知识库中都找不到对应内容"
                f"（查询覆盖率 {coverage * 100:.0f}%，低于拒答阈值 {threshold * 100:.0f}%），"
                f"因此按设计拒答，而不是用不相关的片段硬凑一个答案。"
            )
            warnings = [
                f"相关性门控拦截：覆盖率 {coverage:.2f} < {threshold:.2f}，已拒答"
            ]
        return {
            "query": query,
            "answer": text,
            "mode": "no_answer",
            "mode_label": "检索无命中（已拒答）",
            "model": "—",
            "citations": [],
            "hits": [],
            "warnings": warnings,
            "context": "",
            "elapsed_ms": int((time.time() - started) * 1000),
        }

    context = build_context(hits, cfg.max_context_chars)
    mode = "llm"
    model = cfg.llm_model
    degraded = ""
    if cfg.llm_enabled:
        try:
            text, model = _call_llm(query, context, cfg)
        except Exception as exc:
            text = _extractive(query, hits)
            mode = "extractive_fallback"
            degraded = f"LLM 调用失败（{type(exc).__name__}: {exc}），已降级为检索式答案。"
    else:
        text = _extractive(query, hits)
        mode = "extractive"
        model = "（本地抽取式，无大模型）"

    citations, warnings = verify_citations(text, hits)
    if degraded:
        warnings.append(degraded)
    return {
        "query": query,
        "answer": text,
        "mode": mode,
        "model": model,
        "mode_label": {"llm": "LLM 生成", "extractive": "本地抽取式",
                       "extractive_fallback": "抽取式（LLM 降级）"}[mode],
        "citations": citations,
        "hits": [h.to_dict() for h in hits],
        "warnings": warnings,
        "context": context,
        "elapsed_ms": int((time.time() - started) * 1000),
    }
