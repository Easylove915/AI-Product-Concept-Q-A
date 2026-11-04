# -*- coding: utf-8 -*-
"""轻量中英文分词器（零依赖）。

生产环境建议替换为 jieba / tiktoken / 模型 tokenizer。
这里用「英文按词 + 中文单字与二元组」的经典近似方案，
对中文短查询的召回效果明显好于纯单字切分，且不需要下载任何模型。
"""
from __future__ import annotations

import re
from collections import Counter

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_\-\.]{1,}")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")

STOPWORDS = {
    # 英文
    "the", "and", "for", "are", "with", "that", "this", "from", "you", "your",
    "how", "what", "why", "can", "is", "it", "to", "of", "in", "on", "a", "an",
    "be", "as", "at", "by", "or", "not", "do", "does", "if", "we", "our",
    # 中文虚词 / 疑问词 / 泛指词
    "的", "了", "和", "是", "在", "有", "也", "就", "都", "而", "及", "与",
    "或", "一个", "这个", "那个", "我们", "你们", "他们", "什么", "怎么",
    "如何", "为什么", "哪些", "请问", "可以", "是否", "以及", "对于",
    "关于", "通过", "进行", "使用", "一下", "吗", "呢", "啊", "把", "被",
    "我", "你", "他", "它", "这", "那", "其", "中", "为", "上", "下",
    # 疑问 / 泛指的二元组：命中它们不代表任何相关性，必须与单字一起过滤。
    # 漏掉这一条时，「MCP 是什么」会被「什么」这个高频二元组带到无关文档上。
    "多少", "几个", "怎样", "为啥", "干嘛", "还有", "就是", "这样", "那样",
    "这种", "那种", "一下", "一直", "然后", "而且", "但是", "如果", "因为",
    "所以", "时候", "东西", "问题", "情况", "方面", "内容", "地方",
}


def _normalize(text: str) -> str:
    """全角转半角 + 统一小写。"""
    out = []
    for ch in text.lower():
        code = ord(ch)
        if code == 0x3000:
            code = 32
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        out.append(chr(code))
    return "".join(out)


def tokenize(text: str) -> list[str]:
    """把文本切成检索用的 token 列表（保留重复）。"""
    normalized = _normalize(text or "")
    tokens: list[str] = []

    for match in _WORD_RE.finditer(normalized):
        word = match.group(0).strip("-.")
        if len(word) >= 2 and word not in STOPWORDS:
            tokens.append(word)
        # 长英文词额外保留词干前缀，缓解单复数/时态差异
        if len(word) > 6:
            tokens.append(word[: max(4, len(word) - 2)])

    for match in _CJK_RE.finditer(normalized):
        seg = match.group(0)
        # 单字
        tokens.extend(ch for ch in seg if ch not in STOPWORDS)
        # 二元组，中文检索的主力特征。这里同样要过停用词表：
        # 「什么」「怎么」这类疑问词在语料里到处都是，不过滤会把检索直接带偏。
        tokens.extend(
            gram for gram in (seg[i : i + 2] for i in range(len(seg) - 1))
            if gram not in STOPWORDS
        )

    # 查询恰好全由停用词组成（如 "what is this"）时，回退到不过滤的切分，
    # 否则检索层会直接返回空结果。
    if not tokens and normalized.strip():
        for match in _WORD_RE.finditer(normalized):
            tokens.append(match.group(0).strip("-."))
        for match in _CJK_RE.finditer(normalized):
            seg = match.group(0)
            tokens.extend(seg)
            tokens.extend(seg[i : i + 2] for i in range(len(seg) - 1))

    return tokens


def term_freq(text: str) -> Counter:
    """词频，二元中文词给 1.5 倍权重以减少单字噪声。"""
    tf: Counter = Counter()
    for token in tokenize(text):
        weight = 1.5 if len(token) == 2 and all("\u4e00" <= c <= "\u9fff" for c in token) else 1.0
        tf[token] += weight
    return tf
