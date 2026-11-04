# -*- coding: utf-8 -*-
"""命令行调试入口（不看界面也能验证整条链路）。

    python scripts/cli.py status
    python scripts/cli.py sync --repo owner/repo --path docs
    python scripts/cli.py search "限流阈值"
    python scripts/cli.py ask "429 怎么排查？"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.pipeline import SERVICE  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 知识库 Demo 命令行工具")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="查看索引状态")
    p_sync = sub.add_parser("sync", help="重新拉取文档并重建索引")
    p_sync.add_argument("--repo", default="", help="临时指定 GitHub 仓库")
    p_sync.add_argument("--branch", default="", help="临时指定分支")
    p_sync.add_argument("--path", default="", help="临时指定子目录")
    p_search = sub.add_parser("search", help="只做检索，不生成答案")
    p_search.add_argument("query")
    p_ask = sub.add_parser("ask", help="完整问答")
    p_ask.add_argument("query")
    p_ask.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    if args.cmd == "status":
        s = SERVICE.status()
        print(f"文档 {s.get('num_documents', 0)} 篇 / 片段 {s.get('num_chunks', 0)} 个")
        print(f"来源 {s.get('github_repo') or s.get('docs_dir')}")
        print(f"模式 {s.get('mode')} / 稠密检索 {s.get('dense_mode')}")
        print(f"索引 {s.get('index_path')}")
        for doc in s.get("documents", []):
            print(f"  - {doc['path']}  ({doc['chars']} 字符)")
        return

    if args.cmd == "sync":
        result = SERVICE.sync(
            {"github_repo": args.repo, "github_branch": args.branch, "github_path": args.path}
        )
        print(f"✓ {result['num_documents']} 篇文档 → {result['num_chunks']} 个片段")
        print(f"  索引已写入 {result['index_path']}")
        return

    if args.cmd == "search":
        result = SERVICE.search(args.query)
        if result.get("error"):
            print(result["error"])
            return
        print(f"查询：{result['query']}  命中 {len(result['hits'])} 个片段")
        for hit in result["hits"]:
            c = hit["chunk"]
            print(f"\n[{hit['rank']}] {c['doc_path']} › {c['heading']}  "
                  f"(RRF {hit['rrf_score']:.4f} / BM25 {hit['bm25']:.2f} / TF-IDF {hit['tfidf']:.3f})")
            print("    " + c["content"][:160].replace("\n", " "))
        return

    result = SERVICE.ask(args.query, args.top_k)
    print("\n" + "=" * 68)
    print(f"问题：{result['query']}")
    print(f"模式：{result.get('mode_label', result['mode'])}  "
          f"耗时：{result['elapsed_ms']} ms")
    print("=" * 68)
    print(result["answer"])
    print("\n" + "-" * 68)
    print(f"引用 {len(result['citations'])} 条：")
    for c in result["citations"]:
        print(f"  [{c['index']}] {c['doc_path']} L{c['line_start']}-L{c['line_end']}  {c['heading']}")
        if c["github_url"]:
            print(f"       {c['github_url']}")
    for warning in result["warnings"]:
        print(f"  ⚠ {warning}")


if __name__ == "__main__":
    main()
