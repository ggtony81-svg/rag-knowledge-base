# -*- coding: utf-8 -*-
"""
分块与截断分析

回答论文里必须交代清楚的一个问题：**为什么通用向量模型会在检索上输给 BM25？**

做法是把 934 篇评测语料逐篇过一遍分词器，统计它们的真实 token 长度，
再分别看两种分块策略下的情况：

    整篇直接编码  —— 超过 512 token 的部分被模型静默丢弃（不报错、不警告）
    smart_chunk   —— 只按"第X章/第X条"切，切完仍然超长的块还是会被截断
    smart_chunk_v2—— 结构切 + 按句边界的长度切，目标是让每一块都进得了 512

输出 eval/results/chunking_analysis.json，供论文与绘图脚本引用。

用法：
    python eval/analyze_truncation.py
"""
import json
import os
import sys
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import load_corpus                      # noqa: E402
from text_chunk import (MODEL_MAX_TOKENS, get_tokenizer,  # noqa: E402
                        n_tokens, smart_chunk, smart_chunk_v2)

RESULT_DIR = os.path.join(BASE_DIR, "eval", "results")


def _pct(vals, p):
    """简单分位数（最近秩法），样本量小，不值得引 numpy。"""
    if not vals:
        return 0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def main():
    tokenizer = get_tokenizer()
    corpus = load_corpus(dedup=True)
    texts = corpus["text"].tolist()
    n_doc = len(texts)
    print(f"语料 {n_doc} 篇，开始逐篇统计 token 长度 ...")

    # ---- 1. 整篇长度分布 ----
    raw_lens = []
    for i, t in enumerate(texts):
        raw_lens.append(n_tokens(t, tokenizer))
        if (i + 1) % 200 == 0:
            print(f"  已处理 {i + 1}/{n_doc}")

    over = [n for n in raw_lens if n > MODEL_MAX_TOKENS]
    whole = {
        "n_docs": n_doc,
        "max_tokens": MODEL_MAX_TOKENS,
        "len_mean": round(sum(raw_lens) / n_doc, 1),
        "len_median": _pct(raw_lens, 50),
        "len_p90": _pct(raw_lens, 90),
        "len_p95": _pct(raw_lens, 95),
        "len_max": max(raw_lens),
        "n_over_limit": len(over),
        "pct_over_limit": round(100 * len(over) / n_doc, 1),
        # 被丢掉的那部分平均占全文多少（衡量损失有多大）
        "wasted_token_ratio": round(
            sum(n - MODEL_MAX_TOKENS for n in over) / sum(raw_lens), 4),
    }
    print(f"整篇编码：{whole['pct_over_limit']}% 的文档超过 {MODEL_MAX_TOKENS} token，会被截断")

    # ---- 2. 方案 E/F 用的分块策略 ----
    strategies = {}
    for name, fn, kw in [
        ("smart_chunk", smart_chunk, {}),
        ("smart_chunk_v2", smart_chunk_v2,
         {"max_tokens": 400, "overlap_tokens": 50}),
    ]:
        chunks_per_doc = []
        chunk_lens = []
        for t in texts:
            cs = fn(t, **kw)
            chunks_per_doc.append(len(cs))
            for c in cs:
                chunk_lens.append(n_tokens(c, tokenizer))

        n_chunk = len(chunk_lens)
        n_over = sum(1 for n in chunk_lens if n > MODEL_MAX_TOKENS)
        strategies[name] = {
            "n_chunks": n_chunk,
            "chunks_per_doc_mean": round(n_chunk / n_doc, 2),
            "chunks_per_doc_max": max(chunks_per_doc),
            "chunk_len_mean": round(sum(chunk_lens) / n_chunk, 1) if n_chunk else 0,
            "chunk_len_max": max(chunk_lens) if chunk_lens else 0,
            "n_over_limit": n_over,
            "pct_chunks_truncated": round(100 * n_over / n_chunk, 2) if n_chunk else 0,
            # 一篇文档一个块都没切出来（整篇当作一块）的比例
            "pct_doc_unchunked": round(
                100 * sum(1 for c in chunks_per_doc if c <= 1) / n_doc, 1),
        }
        s = strategies[name]
        print(f"{name}: {s['n_chunks']} 块，每篇 {s['chunks_per_doc_mean']} 块，"
              f"平均 {s['chunk_len_mean']} token，最大 {s['chunk_len_max']}，"
              f"截断率 {s['pct_chunks_truncated']}%")

    out = {
        "dataset": "CMRC2018 检索评测语料（去重后）",
        "tokenizer": "BAAI/bge-small-zh-v1.5",
        "whole_doc": whole,
        "strategies": strategies,
        "raw_lengths": raw_lens,          # 供绘图脚本画直方图
        "note": (
            "整篇编码时超出 512 token 的部分会被模型静默丢弃——不抛异常、不打警告，"
            "检索结果照常返回，只是拿不到后半篇的内容。"
            "文中'截断率'指该策略下仍然超过 512 token 的块所占比例。"
        ),
    }

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, "chunking_analysis.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已保存：{p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
