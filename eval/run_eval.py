# -*- coding: utf-8 -*-
"""RAG 知识库评测脚本。

两种口径，分开测，别混在一起：

    python eval/run_eval.py            # 只测检索层（快、不花钱、调参用）
    python eval/run_eval.py --full     # 测完整链路（含 LLM 生成与拒答）

为什么要分开：调 chunk 大小、改 rerank 权重这类事，只影响检索。
如果每次都等 LLM 跑 30 题，一轮要两三分钟还给 API 交钱，
调参根本跑不动。检索层的 Recall 稳定了，再去看端到端。

指标口径

  非拒答题（A/B/C 三类，24 题）
    Hit@1      第一名是否命中 gold 文档
    Recall@K   top-K 里是否出现过任一 gold 文档（K 默认取 TOP_K）
    MRR@10     第一个命中文档的排名倒数，衡量排序质量
    误杀率      本该答却被拒的比例 —— 这个比 Recall 更该盯

  拒答题（D 类，6 题）
    拒答准确率  正确地拒绝回答的比例
    漏放率      把无关问题当正常问题回答了的比例

为什么单独强调「误杀率」：对知识库问答，用户问了个明明在库里的问题却被告知
「没有相关内容」，比答案不够好难解释得多。召回差是能力问题，误杀是信任问题。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from config import CFG  # noqa: E402
from rag.pipeline import SERVICE  # noqa: E402

TESTSET = Path(__file__).with_name("testset.csv")
RESULTS_DIR = Path(__file__).with_name("results")

# 拒答判据：这里刻意自己写一份，不 import rag/generate.py 里的 _NO_INFO_RE。
# 如果评测脚本复用被测代码的判据，那改正则就能改分数 —— 等于自己给自己发合格证。
REJECT_HINTS = (
    "知识库中没有",
    "知识库中未",
    "知识库里没有",
    "没有找到相关",
    "未找到相关",
    "没有收录",
    "不在知识库",
    "无法回答该问题",
)


def _flag(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "y", "是")


def load_testset(path: Path = TESTSET) -> list[dict]:
    # 用 utf-8-sig：CSV 若带 BOM，首列名会变成 \ufeffid，整表会被静默跳过。
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = []
        for row in csv.DictReader(fh):
            if not row.get("id"):
                continue
            rows.append(
                {
                    "id": row["id"].strip(),
                    "type": row["type"].strip(),
                    "query": row["query"].strip(),
                    "gold": [g.strip() for g in (row.get("gold") or "").split("|") if g.strip()],
                    "expect_reject": _flag(row.get("expect_reject", "0")),
                    "note": (row.get("note") or "").strip(),
                }
            )
    return rows


def _titles_from_hits(hits) -> list[str]:
    """兼容两种返回结构：Hit 对象（检索模式）与 dict（端到端模式）。"""
    out = []
    for hit in hits:
        if isinstance(hit, dict):
            chunk = hit.get("chunk") or {}
            out.append(chunk.get("doc_title", ""))
        else:
            out.append(getattr(hit.chunk, "doc_title", ""))
    return out


def looks_rejected(mode: str, answer_text: str, citations: list) -> bool:
    """判断这次回答是否属于「明确拒答」。

    判据是「开头就在拒答」，而不是「没有引用」。
    最初写的是「命中拒答措辞 and 无引用」，但实测会把一种很好的拒答判错：

        知识库中没有找到关于郑州天气的说明。知识库中仅提供了北京明天天气的示例调用流程 [2][3][4]……

    这种回答为了说明「片段里只有北京的示例」而引用了片段，语义上完全正确，
    却被「有引用 = 在作答」的规则误判。真正该看的是主导意图：
    正常作答不会以「知识库中没有」开头，而以它开头的答案就是在拒答。
    """
    if mode == "no_answer":
        return True
    head = (answer_text or "").strip()[:60]
    return any(hint in head for hint in REJECT_HINTS)


def _summarize(rows: list[dict], gate_aware: bool = True) -> dict:
    """按题目类型聚合。gate_aware=False 时拒答类指标记 None（检索层本就没有拒答能力）。"""
    ok = [r for r in rows if not r["expect_reject"]]
    neg = [r for r in rows if r["expect_reject"]]

    def rate(items, key):
        return round(sum(1 for i in items if i.get(key)) / len(items), 4) if items else 0.0

    reject_accuracy = rate(neg, "rejected") if gate_aware else None
    return {
        "positive_total": len(ok),
        "negative_total": len(neg),
        "hit@1": rate(ok, "hit@1"),
        "recall@k": rate(ok, "recall"),
        "mrr@10": round(sum(i["rr"] for i in ok) / len(ok), 4) if ok else 0.0,
        "false_reject_rate": rate(ok, "rejected"),
        "reject_accuracy": reject_accuracy,
        "leak_rate": (round(1 - reject_accuracy, 4) if reject_accuracy is not None else None),
    }


def run_retrieval(testset: list[dict], top_k: int, gate: bool) -> list[dict]:
    rows = []
    for case in testset:
        started = time.perf_counter()
        hits = SERVICE.index.search(case["query"], top_k=top_k, gate=gate)
        elapsed = int((time.perf_counter() - started) * 1000)
        titles = _titles_from_hits(hits)
        gold = set(case["gold"])

        ranks = [i + 1 for i, t in enumerate(titles) if t in gold] if gold else []
        first = min(ranks) if ranks else 0
        # gate 关闭时「返回空」不再是拒答判断依据 —— 检索层本身没有拒答能力
        rejected = bool(gate and not hits)
        rows.append(
            {
                **case,
                "top_titles": titles,
                "first_rank": first,
                "hit@1": first == 1,
                "recall": bool(ranks),
                "rr": round(1 / first, 4) if first else 0.0,
                "rejected": rejected,
                "gate": SERVICE.index.last_gate,
                "elapsed_ms": elapsed,
            }
        )
    return rows


def run_full(testset: list[dict], top_k: int) -> list[dict]:
    rows = []
    for case in testset:
        started = time.perf_counter()
        result = SERVICE.ask(case["query"], top_k=top_k)
        elapsed = int((time.perf_counter() - started) * 1000)
        if "error" in result:
            rows.append({**case, "top_titles": [], "first_rank": 0, "hit@1": False,
                         "recall": False, "rr": 0.0, "rejected": False,
                         "error": result["error"], "elapsed_ms": elapsed})
            continue

        titles = _titles_from_hits(result.get("hits") or [])
        citations = result.get("citations") or []
        gold = set(case["gold"])
        ranks = [i + 1 for i, t in enumerate(titles) if t in gold] if gold else []
        first = min(ranks) if ranks else 0
        cited_titles = [c.get("doc_title", "") for c in citations]

        rows.append(
            {
                **case,
                "top_titles": titles,
                "first_rank": first,
                "hit@1": first == 1,
                "recall": bool(ranks),
                "rr": round(1 / first, 4) if first else 0.0,
                "rejected": looks_rejected(result.get("mode", ""), result.get("answer", ""), citations),
                "mode": result.get("mode", ""),
                "answer": (result.get("answer") or "")[:400],
                "citations": cited_titles,
                "cite_ok": bool(cited_titles) and all(t in gold for t in cited_titles) if gold else None,
                "warnings": result.get("warnings") or [],
                "elapsed_ms": elapsed,
            }
        )
    return rows


def _pct(x: float) -> str:
    return "N/A" if x is None else f"{x * 100:.1f}%"


def render_markdown(summary: dict, rows: list[dict], mode_label: str, config_snapshot: dict) -> str:
    lines = [
        f"# RAG 评测报告 · {mode_label}",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 配置快照",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
    ]
    for key, value in config_snapshot.items():
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## 总览",
        "",
        "| 指标 | 数值 | 说明 |",
        "| --- | --- | --- |",
        f"| Hit@1 | {_pct(summary['hit@1'])} | 第一名就命中的比例 |",
        f"| Recall@K | {_pct(summary['recall@k'])} | top-K 内出现正解的比例 |",
        f"| MRR@10 | {summary['mrr@10']:.3f} | 排序质量，越接近 1 越好 |",
        f"| 误杀率 | {_pct(summary['false_reject_rate'])} | 该答却被拒，越低越好 |",
        f"| 拒答准确率 | {_pct(summary['reject_accuracy'])} | 域外问题正确拒答的比例 |",
        "",
        "## 逐题明细",
        "",
        "| ID | 类型 | 问题 | 期望 | top-3 命中 | 判定 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        expect = "拒答" if r["expect_reject"] else "作答"
        top3 = " / ".join(t or "—" for t in (r.get("top_titles") or [])[:3]) or "（空）"
        if r["expect_reject"]:
            passed = r["rejected"]
        else:
            passed = r["recall"] and not r["rejected"]
        mark = "✅" if passed else "❌"
        if not r["expect_reject"] and r.get("first_rank"):
            mark += f" @{r['first_rank']}"
        lines.append(f"| {r['id']} | {r['type']} | {r['query']} | {expect} | {top3} | {mark} |")

    fails = [
        r for r in rows
        if (r["expect_reject"] and not r["rejected"]) or (not r["expect_reject"] and (not r["recall"] or r["rejected"]))
    ]

    # 域外题的实际回答要能肉眼核对：只打勾叉看不出「拒得对不对」，
    # 比如把无关问题答成一段像模像样但伪造引用的内容，也必须一眼能看出来。
    neg_answered = [r for r in rows if r["expect_reject"] and r.get("answer")]
    if neg_answered:
        lines += ["", "## 域外题实际回答", ""]
        for r in neg_answered:
            mark = "✅ 已拒答" if r["rejected"] else "❌ 未拒答"
            lines.append(f"- **{r['id']}**（{mark}）{r['query']}")
            lines.append(f"  > {(r.get('answer') or '').replace(chr(10), ' ')[:220]}")

    lines += ["", "## 未通过用例", ""]
    if not fails:
        lines.append("无。")
    else:
        for r in fails:
            reason = "该拒没拒" if r["expect_reject"] else ("被误杀" if r["rejected"] else "未召回")
            lines.append(f"- **{r['id']}**（{reason}）：{r['query']}")
            lines.append(f"  - 期望：{r['gold'] or '拒答'}")
            lines.append(f"  - 实际 top5：{r.get('top_titles') or '（空）'}")
            if r.get("note"):
                lines.append(f"  - 考点：{r['note']}")
            if r.get("answer"):
                lines.append(f"  - 答案片段：{r['answer'][:160].replace(chr(10), ' ')}")

    total_ms = sum(r.get("elapsed_ms", 0) for r in rows)
    lines += ["", f"总耗时 {total_ms / 1000:.1f}s，平均每题 {total_ms / max(len(rows), 1):.0f}ms。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 知识库评测")
    parser.add_argument("--full", action="store_true", help="跑完整链路（含 LLM 生成与拒答判定）")
    parser.add_argument(
        "--gate",
        choices=["auto", "on", "off"],
        default="auto",
        help="检索层是否启用相关性门控。auto=跟随生产配置（配了 LLM 就关闭）",
    )
    parser.add_argument("--top-k", type=int, default=CFG.top_k, help=f"检索返回片段数（默认 {CFG.top_k}）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题，调试用")
    parser.add_argument("--only", default="", help="只跑指定 ID 前缀的题，如 D 或 A,B")
    parser.add_argument("--quiet", action="store_true", help="只写报告，不打印逐题结果")
    args = parser.parse_args()

    testset = load_testset()
    if args.only:
        prefixes = tuple(p.strip().upper() for p in args.only.split(",") if p.strip())
        testset = [c for c in testset if c["id"].upper().startswith(prefixes)]
    if args.limit:
        testset = testset[: args.limit]
    if not testset:
        print("评测集为空：请把 eval/testset.example.csv 复制为 eval/testset.csv，"
              "再按格式填入自己的题目（题库不随公开仓库发布）。")
        return

    if not SERVICE.index.chunks:
        print("索引未建立，先执行：python scripts/cli.py sync")
        return

    mode_label = "端到端（含 LLM）" if args.full else "仅检索层"
    gate_on = {"auto": not CFG.llm_enabled, "on": True, "off": False}[args.gate]

    print("=" * 64)
    print(f"  RAG 评测 · {mode_label}")
    print(f"  题目 {len(testset)} 道 · top_k={args.top_k} · 索引 {len(SERVICE.index.chunks)} 片段")
    if not args.full:
        print(f"  相关性门控 {'开启' if gate_on else '关闭'}（--gate {args.gate}）")
    print("=" * 64)

    if args.full:
        rows = run_full(testset, args.top_k)
    else:
        rows = run_retrieval(testset, args.top_k, gate_on)
    summary = _summarize(rows, gate_aware=(True if args.full else gate_on))

    if not args.quiet:
        for r in rows:
            if r["expect_reject"]:
                status = "拒答 ✅" if r["rejected"] else "漏放 ❌"
            else:
                status = f"@{r['first_rank']} ✅" if r["recall"] and not r["rejected"] else (
                    "误杀 ❌" if r["rejected"] else "未召回 ❌"
                )
            print(f"  {r['id']:<4} {status:<10} {r['query'][:34]}")
        print("-" * 64)

    print(f"  Hit@1           {_pct(summary['hit@1'])}")
    print(f"  Recall@{args.top_k}        {_pct(summary['recall@k'])}")
    print(f"  MRR@10          {summary['mrr@10']:.3f}")
    print(f"  误杀率          {_pct(summary['false_reject_rate'])}")
    print(f"  拒答准确率      {_pct(summary['reject_accuracy'])}")
    print("=" * 64)

    config_snapshot = {
        "答案模式": CFG.mode_label,
        "相关性门控": ("由 LLM 判断" if args.full else ("开启" if gate_on else "关闭")),
        "dense_mode": CFG.dense_mode,
        "chunk_size / overlap": f"{CFG.chunk_size} / {CFG.chunk_overlap}",
        "top_k": args.top_k,
        "min_coverage": CFG.min_coverage,
        "llm_model": CFG.llm_model if CFG.llm_enabled else "（未配置）",
        "文档 / 片段": f"{len(SERVICE.index.documents)} / {len(SERVICE.index.chunks)}",
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = "full" if args.full else f"retrieval-gate{args.gate}"
    payload = {
        "mode": tag,
        "generated_at": stamp,
        "config": config_snapshot,
        "summary": summary,
        "rows": rows,
    }
    json_path = RESULTS_DIR / f"{stamp}-{tag}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md = render_markdown(summary, rows, mode_label, config_snapshot)
    md_path = RESULTS_DIR / f"{stamp}-{tag}.md"
    md_path.write_text(md, encoding="utf-8")
    (RESULTS_DIR / f"latest-{tag}.md").write_text(md, encoding="utf-8")

    print(f"  报告  {md_path}")
    print(f"  明细  {json_path}")


if __name__ == "__main__":
    main()
