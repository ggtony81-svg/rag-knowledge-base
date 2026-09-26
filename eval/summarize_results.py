# -*- coding: utf-8 -*-
"""
检索实验结果汇总

把散落在多个 retrieval_*.json 里的各方案结果合并成一份完整表格，
同时从运行日志中补上"方案C 向量+重排序"的结果。

为什么要从日志里补方案C：
    2026-09-21 那次运行把方案C 跑完之后、在准备跑方案C2 时，
    因为访问 HuggingFace 镜像被拒绝而中断退出，程序没有走到保存 JSON 那一步。
    方案C 的指标是真实跑出来的，完整记录在 eval/results/rerank_run.log 里，
    因此这里从日志中提取，而不是重新跑一遍（重排序在 CPU 上约需 80 分钟）。

用法：
    python eval/summarize_results.py
输出：
    eval/results/retrieval_summary.json   机器可读
    eval/results/retrieval_summary.md     论文用表格
"""
import glob
import json
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")

# 评测集查询总数。只有跑满全部查询的运行才被采纳，调试用的少量查询运行一律排除。
FULL_QUERY_COUNT = 1149

# 表格中的排列顺序（按实验逻辑：两个基线 → 重排序 → 两个单项改进 → 完整方案）
SCHEME_ORDER = ["bm25", "dense", "dense_rerank", "dense_ft",
                "dense_chunk", "dense_chunk_ft"]

# 方案名 -> 论文中的编号称呼
DISPLAY = {
    "bm25":           "A",
    "dense":          "B",
    "dense_rerank":   "C",
    "dense_ft":       "D",
    "dense_chunk":    "E",
    "dense_chunk_ft": "F",
}


def load_from_jsons(min_queries=FULL_QUERY_COUNT):
    """
    合并所有 retrieval_*.json。

    按文件修改时间从新到旧读取，同一个方案只取最新一次有效运行的结果，
    这样同一次运行里测出来的方案耗时彼此可比。

    只接受跑满整个评测集的运行：调试阶段用 --limit 跑少量查询时，
    结果波动极大（例如只跑 8 条查询时，某方案的 P@1 等各项指标恰好全等于
    0.875，看似"整齐"，其实没有统计意义），必须排除。
    """
    # 注意排除本脚本自己的输出文件，否则会把上一次的汇总当成一次运行结果读进来
    files = sorted(
        (p for p in glob.glob(os.path.join(RESULT_DIR, "retrieval_*.json"))
         if not os.path.basename(p).startswith("retrieval_summary")),
        key=os.path.getmtime, reverse=True,
    )
    merged, origin, skipped = {}, {}, []
    for path in files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        n = data.get("dataset", {}).get("query_count", 0)
        if n < min_queries:
            skipped.append(f"{os.path.basename(path)}（仅 {n} 条查询）")
            continue
        for scheme, val in data.get("results", {}).items():
            if scheme not in merged:
                merged[scheme] = val
                origin[scheme] = os.path.basename(path)

    for s in skipped:
        print(f"  跳过非全量运行：{s}")
    return merged, origin


def _read_text(path):
    """
    读取文本文件，自动识别编码。

    本项目的日志文件编码不统一：显式设置了 PYTHONIOENCODING=utf-8 的任务
    写出的是 UTF-8，没设置的会按 Windows 本地编码（GBK）写出。
    GBK 中文按 UTF-8 解会失败，因此这里逐个尝试，避免读出乱码导致解析失败。
    """
    raw = open(path, "rb").read()
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# 日志中打印指标的那一行，形如：
#   方案C 向量+重排序   P@1=0.9243  Recall@1=0.9243  ...  search_ms_mean=6345.68
METRIC_LINE = re.compile(
    r"^\s*(\S[^\n]*?)\s+P@1=([\d.]+)\s+Recall@1=([\d.]+)\s+Recall@3=([\d.]+)\s+"
    r"Recall@5=([\d.]+)\s+Recall@10=([\d.]+)\s+MRR@10=([\d.]+)\s+nDCG@10=([\d.]+)\s+"
    r"index_sec=([\d.]+)\s+search_ms_mean=([\d.]+)\s+search_ms_p95=([\d.]+)\s*$",
    re.MULTILINE,
)


def _scheme_of(label):
    """把日志里的方案名映射回方案标识。注意 C2 要先判断，否则会被 C 抢先匹配。"""
    if "C2" in label:
        return "bm25_rerank"
    if "方案C" in label:
        return "dense_rerank"
    return None


def load_from_logs():
    """
    从运行日志中提取结果。

    目前只有方案C 需要走这条路：2026-09-21 那次运行跑完方案C 之后、
    在准备跑方案C2 时因访问 HuggingFace 镜像被拒绝而中断退出，
    程序没走到保存 JSON 那一步。方案C 的指标是真实跑出来的，
    完整记录在运行日志里，因此从日志提取，而不是重跑（重排序在 CPU 上约需 80 分钟）。
    """
    found = {}
    for name in ("rerank_run.log",):
        path = os.path.join(RESULT_DIR, name)
        if not os.path.exists(path):
            continue
        content = _read_text(path)
        for m in METRIC_LINE.finditer(content):
            label = m.group(1).strip()
            scheme = _scheme_of(label)
            if scheme is None or scheme in found:
                continue
            g = m.groups()
            found[scheme] = {
                "label": label,
                "metrics": {"P@1": float(g[1]), "Recall@1": float(g[2]),
                            "Recall@3": float(g[3]), "Recall@5": float(g[4]),
                            "Recall@10": float(g[5]), "MRR@10": float(g[6]),
                            "nDCG@10": float(g[7])},
                "timing": {"index_sec": float(g[8]),
                           "search_ms_mean": float(g[9]),
                           "search_ms_p95": float(g[10])},
                "_from_log": name,
            }
    return found


def main():
    merged, origin = load_from_jsons()

    extra_note = {}
    for scheme, entry in load_from_logs().items():
        if scheme in merged:
            continue
        merged[scheme] = entry
        origin[scheme] = entry["_from_log"]
        note = (f"该方案跑完后程序因网络中断退出，未保存 JSON，指标从运行日志 "
                f"eval/results/{entry['_from_log']} 中提取。")
        if scheme == "dense_rerank":
            note += ("该方案重排序阶段的召回候选数取 10，与最终返回条数相同，"
                     "因此重排序只能改变排序、无法提高召回上限——"
                     "这正是它的 Recall@10 与方案B 完全相同的原因。")
        extra_note[scheme] = note

    rows = []
    for scheme in SCHEME_ORDER:
        if scheme not in merged:
            continue
        v = merged[scheme]
        m, t = v["metrics"], v["timing"]
        rows.append({
            "scheme": scheme,
            "no": DISPLAY.get(scheme, "?"),
            "label": v["label"],
            "P@1": m["P@1"],
            "Recall@1": m["Recall@1"],
            "Recall@3": m["Recall@3"],
            "Recall@5": m["Recall@5"],
            "Recall@10": m["Recall@10"],
            "MRR@10": m["MRR@10"],
            "nDCG@10": m["nDCG@10"],
            "search_ms_mean": t["search_ms_mean"],
            "index_sec": t["index_sec"],
            "source": origin.get(scheme, ""),
        })

    summary = {
        "dataset": {
            "corpus_size": 934,
            "corpus_raw": 1149,
            "query_count": 1149,
            "note": "CMRC2018 的 BEIR 格式检索集（jinaai/longcontext-cmrc2018-zh），"
                    "语料按文本去重 215 篇，每条查询有 1 篇标准答案文档。",
        },
        "rows": rows,
        "notes": extra_note,
        "timing_caveat": (
            "各方案耗时均在同一台机器上实测（6 核 12 线程 CPU，无 GPU，PyTorch CPU 版）。"
            "但并非全部在同一次运行中测得：A/B/D/E 来自同一次运行，F 来自另一次，"
            "C 来自 2026-09-21 那次运行。其中 C 测耗时期间另有训练任务在争抢 CPU，"
            "因此其 6345 ms 偏保守，独占 CPU 时会低于该值。"
            "指标（P@1 等）不受此影响，因为检索结果只取决于模型和算法、与算力无关——"
            "这一点已通过重跑验证：同一次实验重跑后 BM25 与向量检索的全部指标完全一致。"
        ),
    }

    out_json = os.path.join(RESULT_DIR, "retrieval_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 论文用 Markdown 表格
    lines = ["| 方案 | P@1 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | 检索耗时(ms) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['no']} {r['label'].split(' ', 1)[1]} | {r['P@1']:.4f} | {r['Recall@1']:.4f} "
            f"| {r['Recall@3']:.4f} | {r['Recall@5']:.4f} | {r['Recall@10']:.4f} "
            f"| {r['MRR@10']:.4f} | {r['nDCG@10']:.4f} | {r['search_ms_mean']:.1f} |"
        )
    out_md = os.path.join(RESULT_DIR, "retrieval_summary.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print()
    for r in rows:
        print(f"  {r['no']} {r['label']:<22} 来源：{r['source']}")
    print()
    print(f"已保存：{out_json}")
    print(f"已保存：{out_md}")


if __name__ == "__main__":
    main()
