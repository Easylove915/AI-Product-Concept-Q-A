# -*- coding: utf-8 -*-
"""文档接入层：GitHub 仓库 或 本地目录 → 统一的 Document 列表。

这一层只负责"把文档拿进来并变成纯文本"，不关心分块和检索，
所以以后换成 S3 / 数据库 / Confluence 时，下游代码完全不用改。
"""
from __future__ import annotations

import base64
import io
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import Config
from rag.chunker import Document

_API = "https://api.github.com"
_RAW = "https://raw.githubusercontent.com"
_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

SKIP_DIR_PARTS = {
    ".git", "node_modules", "vendor", "dist", "build", "__pycache__",
    ".venv", "venv", "site-packages", ".github", "assets", "images",
}


class IngestError(Exception):
    pass


def parse_repo_spec(spec: str) -> tuple[str, str]:
    """把 'owner/repo' 或 GitHub URL 解析成 (owner, repo)。"""
    spec = spec.strip().rstrip("/")
    if not spec:
        raise IngestError("GITHUB_REPO 为空")
    if spec.startswith("http://") or spec.startswith("https://"):
        parsed = urllib.parse.urlparse(spec)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 2:
            raise IngestError(f"无法从 URL 解析 owner/repo：{spec}")
        return parts[0], parts[1].removesuffix(".git")
    parts = spec.split("/")
    if len(parts) != 2 or not all(parts):
        raise IngestError("GITHUB_REPO 应形如 owner/repo 或完整 GitHub URL")
    return parts[0], parts[1].removesuffix(".git")


def _headers(cfg: Config, json_api: bool = True) -> dict:
    headers = {"User-Agent": "rag-kb-demo/1.0"}
    if json_api:
        headers["Accept"] = "application/vnd.github+json"
    if cfg.github_token:
        headers["Authorization"] = f"Bearer {cfg.github_token}"
    return headers


def _get(url: str, cfg: Config, timeout: int = 30) -> bytes:
    request = urllib.request.Request(url, headers=_headers(cfg))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise IngestError(f"GitHub API {exc.code}: {url}") from exc
    except urllib.error.URLError as exc:
        raise IngestError(f"网络请求失败：{exc.reason}") from exc


def allowed_exts(cfg: Config) -> set[str]:
    return {
        ext.strip().lower() if ext.strip().startswith(".") else "." + ext.strip().lower()
        for ext in cfg.github_exts.split(",")
        if ext.strip()
    }


def _tree(cfg: Config, owner: str, repo: str) -> tuple[str, list[str]]:
    """获取仓库文件树；main 不存在时自动回退 master。"""
    branches = [cfg.github_branch] if cfg.github_branch else ["main", "master"]
    branches += [b for b in ("main", "master") if b not in branches]
    last_error: Exception | None = None
    for branch in branches:
        url = f"{_API}/repos/{owner}/{repo}/git/trees/{urllib.parse.quote(branch, safe='')}?recursive=1"
        try:
            payload = json.loads(_get(url, cfg).decode("utf-8"))
        except IngestError as exc:
            last_error = exc
            continue
        if payload.get("truncated"):
            # 超大仓库 tree 会被截断，此时建议用 git clone（README 有说明）
            print("[ingest] 提示：仓库文件树被截断，部分文件可能未包含")
        paths = [
            item["path"]
            for item in payload.get("tree", [])
            if item.get("type") == "blob" and item.get("path")
        ]
        return branch, paths
    raise IngestError(f"无法读取仓库 {owner}/{repo} 的文件树：{last_error}")


def _paths_in_scope(paths: list[str], cfg: Config) -> list[str]:
    exts = allowed_exts(cfg)
    prefix = cfg.github_path.strip().strip("/")
    result = []
    for path in paths:
        parts = set(path.split("/"))
        if parts & SKIP_DIR_PARTS:
            continue
        if prefix and not path.startswith(prefix + "/"):
            continue
        name = Path(path).name.lower()
        # 无扩展名的 README / README.txt 之类纯文本也纳入知识库
        if Path(path).suffix.lower() not in exts and not name.startswith("readme"):
            continue
        result.append(path)
    return sorted(result)


def _fetch_one(owner: str, repo: str, branch: str, path: str, cfg: Config) -> tuple[str, bytes]:
    """返回 (path, 原始字节)。私有仓库走 API contents + base64。"""
    quoted = urllib.parse.quote(path, safe="/")
    if cfg.github_token:
        url = f"{_API}/repos/{owner}/{repo}/contents/{quoted}?ref={urllib.parse.quote(branch)}"
        payload = json.loads(_get(url, cfg).decode("utf-8"))
        return path, base64.b64decode(payload.get("content", ""))
    url = f"{_RAW}/{owner}/{repo}/{urllib.parse.quote(branch, safe='')}/{quoted}"
    return path, _get(url, cfg)


def _to_text(path: str, raw: bytes) -> str:
    if path.lower().endswith(".docx"):
        return _docx_text(raw)
    return raw.decode("utf-8", errors="replace")


def _docx_text(data: bytes) -> str:
    """用标准库解析 docx（zip + XML），无需 python-docx。"""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ET.ParseError):
        return ""
    lines: list[str] = []
    for para in root.iter(_W_NS + "p"):
        text = "".join(node.text or "" for node in para.iter(_W_NS + "t"))
        if not text.strip():
            continue
        style = para.find(f"{_W_NS}pPr/{_W_NS}pStyle")
        val = style.get(_W_NS + "val", "") if style is not None else ""
        match = re.search(r"Heading(\d)", val)
        lines.append("#" * int(match.group(1)) + " " + text if match else text)
    return "\n\n".join(lines)


_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")


def parse_meta(path: str, text: str) -> tuple[str, list[str], list[str]]:
    """解析 Obsidian 风格文档的元数据 → (标题, 标签, 关联概念)。

    这批知识库每篇都有 YAML front matter（title / tags），正文用 [[双链]]
    互相关联。把这些信息抽出来给分块层用，能显著提升「按概念名提问」的召回。
    """
    title = ""
    tags: list[str] = []
    extra_aliases: list[str] = []
    match = _FRONT_MATTER_RE.match(text)
    if match:
        for line in match.group(1).splitlines():
            key, _, value = line.partition(":")
            key, value = key.strip().lower(), value.strip()
            if key == "title" and value:
                title = value.strip('"').strip("'")
            elif key == "tags" and value:
                inner = value.strip().lstrip("[").rstrip("]")
                tags = [t.strip().strip('"').strip("'") for t in inner.split(",") if t.strip()]
            elif key in ("aliases", "alias") and value:
                inner = value.strip().lstrip("[").rstrip("]")
                extra_aliases = [t.strip().strip('"').strip("'") for t in inner.split(",") if t.strip()]

    if not title:
        heading = re.search(r"^\s{0,3}#\s+(.+)$", text, re.M)
        title = heading.group(1).strip() if heading else Path(path).stem
    title = re.sub(r"[（(].*?[）)]", "", title).strip() or Path(path).stem

    aliases: list[str] = []
    seen = {title}
    for raw in extra_aliases + _WIKILINK_RE.findall(text):
        name = re.sub(r"[（(].*?[）)]", "", raw).strip()
        name = name.split("/")[-1].strip()
        if name and name not in seen and len(name) >= 2:
            seen.add(name)
            aliases.append(name)
    return title, tags, aliases


def fetch_github(cfg: Config) -> list[Document]:
    owner, repo = parse_repo_spec(cfg.github_repo)
    branch, paths = _tree(cfg, owner, repo)
    selected = _paths_in_scope(paths, cfg)
    if not selected:
        raise IngestError(
            f"仓库中没有匹配的文档（扩展名 {sorted(allowed_exts(cfg))}，"
            f"子目录 {cfg.github_path or '全部'}）。请检查 GITHUB_PATH / GITHUB_EXTS。"
        )

    documents: list[Document] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_fetch_one, owner, repo, branch, path, cfg): path
            for path in selected
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                rel_path, raw = future.result()
            except Exception as exc:  # 单文件失败不影响整体
                errors.append(f"{path}: {exc}")
                continue
            text = _to_text(rel_path, raw)
            if not text.strip():
                continue
            title, tags, aliases = parse_meta(rel_path, text)
            documents.append(
                Document(
                    doc_id=f"github:{owner}/{repo}:{rel_path}",
                    path=rel_path,
                    title=title,
                    text=text,
                    source="github",
                    github_url=f"https://github.com/{owner}/{repo}/blob/{branch}/{rel_path}",
                    tags=tags,
                    aliases=aliases,
                )
            )
    if errors:
        print(f"[ingest] {len(errors)} 个文件读取失败，例如：{errors[:2]}")
    documents.sort(key=lambda d: d.path)
    return documents


def load_local(cfg: Config) -> list[Document]:
    root = cfg.docs_dir
    if not root.exists():
        raise IngestError(f"本地文档目录不存在：{root}")
    exts = allowed_exts(cfg)
    documents: list[Document] = []
    for file in sorted(root.rglob("*")):
        if not file.is_file():
            continue
        if file.suffix.lower() not in exts and not file.name.lower().startswith("readme"):
            continue
        if set(file.relative_to(root).parts[:-1]) & SKIP_DIR_PARTS:
            continue
        if file.suffix.lower() == ".docx":
            text = _docx_text(file.read_bytes())
        else:
            text = file.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        rel = file.relative_to(root).as_posix()
        title, tags, aliases = parse_meta(rel, text)
        documents.append(
            Document(
                doc_id=f"local:{rel}",
                path=rel,
                title=title,
                text=text,
                source="local",
                github_url="",
                tags=tags,
                aliases=aliases,
            )
        )
    return documents


def load_documents(cfg: Config) -> list[Document]:
    if cfg.github_repo:
        return fetch_github(cfg)
    return load_local(cfg)
