# -*- coding: utf-8 -*-
"""Web 入口：仅用 Python 标准库启动 HTTP 服务。

    python app.py
    然后浏览器打开 http://127.0.0.1:8765

换成 FastAPI 时，把下面的路由函数原样搬过去即可，业务层完全不用改。
"""
from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from config import CFG, WEB_DIR
from rag.pipeline import SERVICE

MAX_BODY = 1 << 20  # 1 MB


class Handler(BaseHTTPRequestHandler):
    server_version = "RAGKBDemo/1.0"

    # ---------------------------------------------------------------- helpers
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: dict, code: int = 200) -> None:
        self._send(
            code,
            json.dumps(obj, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _static(self, relative: str) -> None:
        target = (WEB_DIR / relative).resolve()
        if not str(target).startswith(str(WEB_DIR.resolve())) or not target.is_file():
            self._json({"error": "not found"}, 404)
            return
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if mime.startswith("text/") or mime in ("application/javascript",):
            mime += "; charset=utf-8"
        self._send(200, target.read_bytes(), mime)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        if length > MAX_BODY:
            raise ValueError("请求体过大")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def log_message(self, fmt: str, *args) -> None:
        print(f"[http] {self.address_string()} - {fmt % args}")

    # ---------------------------------------------------------------- GET
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._static("index.html")
        elif path in ("/style.css", "/app.js", "/favicon.svg"):
            self._static(path.lstrip("/"))
        elif path == "/api/status":
            self._json(SERVICE.status())
        elif path == "/api/docs":
            self._json({"documents": SERVICE.index.documents})
        else:
            self._json({"error": "not found"}, 404)

    # ---------------------------------------------------------------- POST
    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        try:
            body = self._body()
        except Exception as exc:
            self._json({"error": f"请求解析失败：{exc}"}, 400)
            return

        try:
            if path == "/api/sync":
                result = SERVICE.sync(
                    {
                        "github_repo": body.get("repo"),
                        "github_branch": body.get("branch"),
                        "github_path": body.get("path"),
                    }
                )
                self._json(result)
            elif path == "/api/ask":
                self._json(SERVICE.ask(body.get("query", ""), body.get("top_k")))
            elif path == "/api/search":
                self._json(SERVICE.search(body.get("query", ""), body.get("top_k")))
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:
            self._json({"ok": False, "error": str(exc)}, 500)


def main() -> None:
    status = SERVICE.status()
    print("=" * 68)
    print("  AI 知识库答疑系统 Demo  ·  RAG Pipeline")
    print("=" * 68)
    print(f"  地址        http://{CFG.host}:{CFG.port}")
    print(f"  文档来源    {CFG.github_repo or CFG.docs_dir}")
    print(f"  索引        {status.get('num_chunks', 0)} 个片段 / {status.get('num_documents', 0)} 篇文档")
    print(f"  答案模式    {CFG.mode_label}")
    print(f"  稠密检索    {CFG.dense_mode}")
    if status.get("last_error"):
        print(f"  ⚠ {status['last_error']}")
    print("=" * 68)
    server = ThreadingHTTPServer((CFG.host, CFG.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
