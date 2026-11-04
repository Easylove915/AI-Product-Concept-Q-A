# -*- coding: utf-8 -*-
"""全局配置。

所有配置项都有安全默认值：不填任何内容也能跑通 Demo（使用内置示例文档 +
抽取式答案）。接真实数据时只需要改 .env 里的两三项。

配置优先级：环境变量 > 项目根目录 .env 文件 > 默认值
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample_docs"
INDEX_DIR = DATA_DIR / "index"
WEB_DIR = ROOT / "web"


def _load_dotenv(path: Path) -> None:
    """极简 .env 解析器，不依赖 python-dotenv。不覆盖已有的环境变量。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT / ".env")


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


@dataclass
class Config:
    # ---- 文档来源 ---------------------------------------------------------
    # GitHub 仓库：支持 "owner/repo" 或完整 URL；留空则使用本地文档目录
    github_repo: str = field(default_factory=lambda: _env("GITHUB_REPO", ""))
    github_branch: str = field(default_factory=lambda: _env("GITHUB_BRANCH", "main"))
    github_path: str = field(default_factory=lambda: _env("GITHUB_PATH", ""))
    github_token: str = field(default_factory=lambda: _env("GITHUB_TOKEN", ""))
    # 逗号分隔的扩展名白名单
    github_exts: str = field(
        default_factory=lambda: _env("GITHUB_EXTS", ".md,.markdown,.txt,.rst,.docx")
    )
    # 本地文档目录（GITHUB_REPO 为空时生效）
    docs_dir: Path = field(
        default_factory=lambda: Path(_env("DOCS_DIR", str(SAMPLE_DIR))).resolve()
    )

    # ---- 分块 -------------------------------------------------------------
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 600))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 100))
    min_chunk_chars: int = field(default_factory=lambda: _env_int("MIN_CHUNK_CHARS", 40))

    # ---- 检索 -------------------------------------------------------------
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    candidate_k: int = field(default_factory=lambda: _env_int("CANDIDATE_K", 20))
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    # 稠密向量（可选增强）：off | hashing | openai | local
    dense_mode: str = field(default_factory=lambda: _env("DENSE_MODE", "off").lower())
    # 向量模型与批大小。百炼 text-embedding-v3 单次最多 10 条，超了直接 400，
    # 所以批大小做成配置项：换模型（如本地 bge）时不用改代码。
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    embedding_batch: int = field(
        default_factory=lambda: _env_int("EMBEDDING_BATCH", 10)
    )
    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 1024))
    min_score: float = field(default_factory=lambda: _env_float("MIN_SCORE", 0.0))
    # 相关性门控：IDF 加权查询词覆盖率低于该值就判定「知识库里没有」并拒答。
    # 设 0 表示关闭门控（任何问题都会强行返回 top-k）。
    # 0.24 是在这批 wiki 语料（55 篇 / 489 片段）上实测出的安全值：
    # 知识库内问题最低 0.25 全部放行，知识库外问题多数被拦下。
    min_coverage: float = field(
        default_factory=lambda: _env_float("MIN_COVERAGE", 0.24)
    )

    # ---- 生成 -------------------------------------------------------------
    # 兼容 OpenAI /chat/completions 协议：DeepSeek、通义、智谱、Ollama、vLLM 均可
    llm_base_url: str = field(
        default_factory=lambda: _env("LLM_BASE_URL", "https://api.openai.com/v1")
    )
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gpt-4o-mini"))
    llm_temperature: float = field(
        default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.1)
    )
    llm_timeout: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT", 60))
    max_context_chars: int = field(
        default_factory=lambda: _env_int("MAX_CONTEXT_CHARS", 6000)
    )

    # ---- 服务 -------------------------------------------------------------
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8765))

    # ---- 派生 -------------------------------------------------------------
    index_path: Path = field(
        default_factory=lambda: Path(_env("INDEX_PATH", str(INDEX_DIR / "kb_index.json")))
    )

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def mode_label(self) -> str:
        return "LLM 生成" if self.llm_enabled else "本地抽取式（未配置 API Key）"


CFG = Config()
