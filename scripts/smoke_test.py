# -*- coding: utf-8 -*-
"""无 Key 冒烟测试：起一个干净的服务实例 → 查状态 → 问一个问题 → 断言链路通 → 关掉。

    python scripts/smoke_test.py

不需要 API Key、不需要联网、不碰你本地的索引 —— 它用临时索引 + 内置示例文档，
所以无论在谁的机器上跑，结果都一样。CI 里跑的就是这个。

退出码 0 表示通过，非 0 表示失败（CI 会因此变红）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8799  # 刻意避开默认 8765，免得和本地正在跑的服务撞端口
BASE = f"http://127.0.0.1:{PORT}"


def _get(path: str, timeout: float = 3) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as r:
        return json.load(r)


def _post(path: str, payload: dict, timeout: float = 60) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def wait_ready(timeout: float = 120) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            return _get("/api/status")
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last = exc
            time.sleep(0.5)
    raise SystemExit(f"[FAIL] 服务在 {timeout:.0f}s 内没有就绪：{last}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="rag_smoke_"))
    env = {
        **os.environ,
        "PORT": str(PORT),
        # 强制降级：没有 Key 时走抽取式答案，不发起任何网络请求
        "LLM_API_KEY": "",
        "DENSE_MODE": "off",
        # 用临时索引 + 内置示例文档，保证结果可复现
        "INDEX_PATH": str(tmp / "smoke_index.json"),
        "DOCS_DIR": str(ROOT / "data" / "sample_docs"),
    }

    proc = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        status = wait_ready()
        docs = status.get("num_documents", 0)
        chunks = status.get("num_chunks", 0)
        if docs <= 0 or chunks <= 0:
            raise SystemExit(f"[FAIL] 索引没建起来：{docs} 篇 / {chunks} 片段")
        print(f"[ok] 索引就绪：{docs} 篇文档 → {chunks} 个片段")

        if "LLM" in status.get("mode", ""):
            raise SystemExit(f"[FAIL] CI 环境不该有 API Key，但模式是 {status['mode']}")
        print(f"[ok] 答案模式：{status['mode']}")

        result = _post("/api/ask", {"query": "429 错误怎么排查？"})
        if not result.get("answer"):
            raise SystemExit(f"[FAIL] /api/ask 没有返回答案：{result}")
        hits = result.get("hits", [])
        if not hits:
            raise SystemExit("[FAIL] /api/ask 没有召回任何片段")
        if not result.get("citations"):
            raise SystemExit("[FAIL] /api/ask 没有返回引用 —— 溯源链路断了")
        print(f"[ok] 问答链路：{len(hits)} 个片段 / {len(result['citations'])} 条引用 / {result['elapsed_ms']} ms")
        print(f"[ok] 首位命中：{hits[0]['chunk']['doc_path']} › {hits[0]['chunk']['heading']}")

        print("\n冒烟测试通过。")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
