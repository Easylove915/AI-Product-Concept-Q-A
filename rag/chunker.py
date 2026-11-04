# -*- coding: utf-8 -*-
"""文档分块：标题感知 + 段落滑窗 + 重叠。

为什么分块要带标题：RAG 里最容易丢的信息就是"这段话属于哪一节"。
把标题路径拼进 chunk 文本，检索命中和答案引用都会明显更准。

每个 chunk 都携带完整的溯源坐标：文件路径、行号、原文字符区间、
GitHub 在线链接行锚点。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_FRONT_MATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.S)
# 纯导航型章节：全是 [[双链]]，没有实质内容，留着会污染检索
_SKIP_SECTION_RE = re.compile(r"相关页面|相关链接|相关文档|延伸阅读|参见|导航|see also", re.I)


@dataclass
class Document:
    doc_id: str
    path: str                 # 仓库内相对路径 / 本地相对路径
    title: str
    text: str
    source: str               # github | local
    github_url: str = ""      # 文档级别的在线链接
    tags: list[str] = field(default_factory=list)     # front matter 里的标签
    aliases: list[str] = field(default_factory=list)  # [[双链]] 指向的关联概念名


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_path: str
    doc_title: str
    github_url: str           # 带 #Lstart-Lend 行锚点
    heading: str
    text: str                 # 参与检索与送入 LLM 的文本（含标题前缀）
    content: str              # 纯正文，用于前端原文展示
    char_start: int
    char_end: int
    line_start: int
    line_end: int
    order: int

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "Chunk":
        return Chunk(**data)


def _strip_front_matter(text: str) -> tuple[str, int]:
    """移除 YAML front matter，返回 (正文, 被移除的字符数)，保持偏移量对齐。"""
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return text, 0
    return text[match.end() :], match.end()


def _find_break(text: str, start: int, ideal_end: int, hard_end: int) -> int:
    """在理想切点附近找最近的句子/换行边界，避免把一句话拦腰截断。"""
    if ideal_end >= hard_end:
        return hard_end
    window = text[ideal_end : min(hard_end, ideal_end + 120)]
    for mark in ("\n\n", "。", "！", "？", ". ", "\n", "；", "; ", "，", ", "):
        pos = window.find(mark)
        if pos != -1:
            return ideal_end + pos + len(mark)
    return ideal_end


def _windows(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """返回 [(start, end)] 相对偏移的滑窗区间。"""
    if not text.strip():
        return []
    length = len(text)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(length, start + size)
        end = _find_break(text, start, end, length)
        if end <= start:
            end = min(length, start + size)
        spans.append((start, end))
        if end >= length:
            break
        next_start = max(start + 1, end - overlap)
        start = next_start
    return spans


def _sections(text: str, offset: int) -> list[tuple[str, int, str]]:
    """按 Markdown 标题切成 [(标题路径, 全局起始偏移, section 正文)]。"""
    result: list[tuple[str, int, str]] = []
    stack: dict[int, str] = {}
    current_heading = ""
    buf: list[str] = []
    buf_start = offset

    lines = text.splitlines(keepends=True)
    cursor = offset
    for line in lines:
        match = _HEADING_RE.match(line.rstrip("\r\n"))
        if match:
            if buf and "".join(buf).strip():
                raw = "".join(buf)
                leading = len(raw) - len(raw.lstrip("\n"))
                result.append((current_heading, buf_start + leading, raw.strip("\n")))
            level = len(match.group(1))
            title = match.group(2).strip()
            stack = {k: v for k, v in stack.items() if k < level}
            stack[level] = title
            current_heading = " > ".join(stack[k] for k in sorted(stack))
            buf = []
            buf_start = cursor + len(line)
        else:
            buf.append(line)
        cursor += len(line)

    if buf and "".join(buf).strip():
        raw = "".join(buf)
        leading = len(raw) - len(raw.lstrip("\n"))
        result.append((current_heading, buf_start + leading, raw.strip("\n")))
    return result


def chunk_document(
    doc: Document,
    chunk_size: int = 600,
    overlap: int = 100,
    min_chars: int = 40,
) -> list[Chunk]:
    """把一篇文档切成带溯源元数据的 chunk 列表。"""
    text, stripped = _strip_front_matter(doc.text)
    chunks: list[Chunk] = []
    sections = _sections(text, stripped) or [("", stripped, text)]

    for heading, sec_start, section_text in sections:
        # Obsidian 知识库末尾常有一节「相关页面」，内容全是 [[双链]]，
        # 检索时会被概念名大量命中却答不出东西，直接跳过。
        leaf = heading.split(" > ")[-1] if heading else ""
        if leaf and _SKIP_SECTION_RE.search(leaf):
            continue
        # 概念名（文件名 / front matter title）必须进入检索文本。
        # 否则用户问「Agent五要素是什么」，正文里未必出现这五个字，会漏召回。
        full_heading = f"{doc.title} › {heading}" if heading else doc.title
        for rel_start, rel_end in _windows(section_text, chunk_size, overlap):
            content = section_text[rel_start:rel_end].strip()
            if len(content) < min_chars:
                continue
            abs_start = sec_start + rel_start
            abs_end = sec_start + rel_end
            line_start = doc.text.count("\n", 0, abs_start) + 1
            line_end = doc.text.count("\n", 0, abs_end) + 1

            prefix = f"[{full_heading}]\n"
            # 标签与双链别名只放在首个片段，作为检索提示，不污染每一块正文
            if not chunks and (doc.tags or doc.aliases):
                hints = []
                if doc.tags:
                    hints.append("标签：" + "、".join(doc.tags[:8]))
                if doc.aliases:
                    hints.append("相关概念：" + "、".join(doc.aliases[:6]))
                prefix += " ".join(hints) + "\n"
            payload = prefix + content
            raw_id = f"{doc.doc_id}|{abs_start}|{abs_end}"
            chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]

            url = doc.github_url
            if url and "#L" not in url:
                url = f"{url}#L{line_start}-L{line_end}"

            chunks.append(
                Chunk(
                    id=chunk_id,
                    doc_id=doc.doc_id,
                    doc_path=doc.path,
                    doc_title=doc.title,
                    github_url=url,
                    heading=full_heading,
                    text=payload,
                    content=content,
                    char_start=abs_start,
                    char_end=abs_end,
                    line_start=line_start,
                    line_end=line_end,
                    order=len(chunks),
                )
            )
    return chunks
