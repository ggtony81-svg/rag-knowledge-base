# -*- coding: utf-8 -*-
"""
检索方案对比实验

在 CMRC2018 检索评测集上，对四种检索方案做统一评测，输出可用于论文的指标表格。

用法：
    python eval/run_retrieval_eval.py                      # 跑全部方案
    python eval/run_retrieval_eval.py --schemes bm25 dense # 只跑指定方案

结果保存到 eval/results/retrieval_<时间戳>.json，同时打印表格。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset
import metrics
from retrievers import (BM25Retriever, DenseRetriever, RerankRetriever,
                        ChunkedDenseRetriever, EMBED_MODEL_NAME)

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def get_doc_vectors(corpus, model_name=EMBED_MODEL_NAME, force=False):
    """
    计算（并缓存）语料库向量。

    向量只跟"语料 + 模型"有关，与检索方案无关，因此四种方案共用同一份，
    既省时间，也保证方案之间唯一的差别就是检索策略本身。
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    # 缓存文件名要能同时兼容 HuggingFace 模型名（形如 BAAI/bge-small-zh-v1.5）
    # 和本地模型目录（形如 d:/shuqi/eval/cache/finetuned_model）。
    # 后者含 ":" 与 "/"，在 Windows 上是非法文件名字符，必须统一替换掉。
    tag = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in model_name)
    vec_path = os.path.join(CACHE_DIR, f"docvecs_{tag}.npy")
    id_path = os.path.join(CACHE_DIR, f"docids_{tag}.json")

    if os.path.exists(vec_path) and os.path.exists(id_path) and not force:
        with open(id_path, encoding="utf-8") as f:
            cached_ids = json.load(f)
        if cached_ids == list(corpus["id"]):
            return np.load(vec_path)

    print(f"  正在编码 {len(corpus)} 篇文档（首次运行需要一些时间）...")
    t0 = time.time()
    retriever = DenseRetriever(model_name=model_name)
    vecs = retriever.encode_docs(corpus["text"].tolist())
    print(f"  编码完成，耗时 {time.time() - t0:.1f} 秒，向量维度 {vecs.shape[1]}")

    np.save(vec_path, vecs)
    with open(id_path, "w", encoding="utf-8") as f:
        json.dump(list(corpus["id"]), f)
    return vecs


def run_scheme(name, retriever, corpus, queries, qrels, topk=10, prefix=""):
    """跑一个检索方案，返回 (指标, 耗时统计)。"""
    doc_ids = corpus["id"].tolist()
    doc_texts = corpus["text"].tolist()

    t0 = time.time()
    if isinstance(retriever, BM25Retriever):
        # BM25 建的是倒排索引，与文档向量无关
        retriever.index(doc_ids, doc_texts)
    else:
        retriever.index(doc_ids, doc_texts, doc_vecs=corpus.attrs.get("doc_vecs"))
    index_time = time.time() - t0

    qrels_map = qrels.groupby("qid")["pid"].apply(set).to_dict()

    all_ranked = {}
    latencies = []
    for _, row in queries.iterrows():
        t1 = time.time()
        hits = retriever.search(row["text"], topk=topk)
        latencies.append((time.time() - t1) * 1000.0)   # 毫秒
        all_ranked[row["id"]] = [d for d, _ in hits]

    res = metrics.evaluate(all_ranked, qrels_map)
    timing = {
        "index_sec": round(index_time, 2),
        "search_ms_mean": round(float(np.mean(latencies)), 2),
        "search_ms_p95": round(float(np.percentile(latencies, 95)), 2),
    }
    return res, timing, all_ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="*",
                    default=["bm25", "dense", "dense_rerank", "bm25_rerank"],
                    help="要运行的方案：bm25 / dense / dense_rerank / bm25_rerank / "
                         "dense_chunk / dense_ft / dense_chunk_ft")
    ap.add_argument("--chunk-tokens", type=int, default=400,
                    help="dense_chunk 方案的分块长度上限（token）")
    ap.add_argument("--chunk-overlap", type=int, default=50,
                    help="dense_chunk 方案相邻块之间的重叠长度（token）")
    ap.add_argument("--ft-model", default=os.path.join(CACHE_DIR, "finetuned_model"),
                    help="微调后模型所在目录（dense_ft 方案用）")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="只用前 N 条查询（调试用）")
    ap.add_argument("--recall-k", type=int, default=50,
                    help="重排序前向量召回的候选数（CPU 上每对约 0.4 秒，越大越慢）")
    args = ap.parse_args()

    print("=" * 78)
    print("检索方案对比实验 — 数据集：CMRC2018 / BEIR 检索格式")
    print("=" * 78)

    corpus, queries, qrels = dataset.build_retrieval_benchmark(dedup=True)
    if args.limit:
        queries = queries.head(args.limit).reset_index(drop=True)
        qrels = qrels[qrels["qid"].isin(set(queries["id"]))].reset_index(drop=True)

    print(f"语料库：{len(corpus)} 篇文档（原始 {corpus.attrs['n_raw']} 篇，去重 {corpus.attrs['n_raw'] - len(corpus)} 篇）")
    print(f"查询集：{len(queries)} 条，每条 1 篇标准答案文档")
    print()

    need_vectors = any(s in args.schemes for s in ["dense", "dense_rerank", "dense_ft"])
    if need_vectors:
        vecs = get_doc_vectors(corpus)
        corpus.attrs["doc_vecs"] = vecs
    print()

    results = {}
    ranked_all = {}

    for scheme in args.schemes:
        if scheme == "bm25":
            label, r = "方案A BM25关键词", BM25Retriever()
        elif scheme == "dense":
            label, r = "方案B 向量检索", DenseRetriever()
        elif scheme == "dense_rerank":
            label, r = "方案C 向量+重排序", RerankRetriever(base_kind="dense",
                                                          recall_k=args.recall_k)
        elif scheme == "bm25_rerank":
            label, r = "方案C2 BM25+重排序", RerankRetriever(base_kind="bm25",
                                                            recall_k=args.recall_k)
        elif scheme == "dense_chunk":
            # 分块方案自己编码"块"，不依赖文档级别的向量缓存
            label, r = "方案E 分块向量检索", ChunkedDenseRetriever(
                max_tokens=args.chunk_tokens, overlap_tokens=args.chunk_overlap)
        elif scheme == "dense_chunk_ft":
            if not os.path.isdir(args.ft_model):
                print(f"{'方案F 分块+微调':<20} 跳过：微调模型不存在（{args.ft_model}）")
                continue
            label, r = "方案F 分块+微调", ChunkedDenseRetriever(
                model_name=args.ft_model,
                max_tokens=args.chunk_tokens, overlap_tokens=args.chunk_overlap)
        elif scheme == "dense_ft":
            if not os.path.isdir(args.ft_model):
                print(f"{'方案D 向量检索(微调)':<20} 跳过：微调模型不存在（{args.ft_model}）")
                continue
            label, r = "方案D 向量检索(微调)", DenseRetriever(model_name=args.ft_model)
            corpus.attrs["doc_vecs"] = get_doc_vectors(corpus, model_name=args.ft_model)
        else:
            print(f"未知方案：{scheme}")
            continue

        print(f"运行 {label} ...")
        res, timing, ranked = run_scheme(scheme, r, corpus, queries, qrels, topk=args.topk)
        results[scheme] = {"label": label, "metrics": res, "timing": timing}
        ranked_all[scheme] = ranked
        print("   " + metrics.format_result(label, res, timing))
        print()

    # ---------------- 汇总表 ----------------
    print("=" * 78)
    print("汇总")
    print("=" * 78)
    header = f"{'方案':<22}{'P@1':>8}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'R@10':>8}{'MRR@10':>9}{'nDCG@10':>9}{'耗时(ms)':>11}"
    print(header)
    print("-" * len(header))
    for s, v in results.items():
        m, t = v["metrics"], v["timing"]
        print(f"{v['label']:<20}"
              f"{m.get('P@1', 0):>8.4f}{m.get('Recall@1', 0):>8.4f}{m.get('Recall@3', 0):>8.4f}"
              f"{m.get('Recall@5', 0):>8.4f}{m.get('Recall@10', 0):>8.4f}{m.get('MRR@10', 0):>9.4f}"
              f"{m.get('nDCG@10', 0):>9.4f}{t['search_ms_mean']:>11.1f}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULT_DIR, f"retrieval_{stamp}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": {
                "corpus_size": len(corpus),
                "corpus_raw": int(corpus.attrs["n_raw"]),
                "query_count": len(queries),
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存：{out}")


if __name__ == "__main__":
    main()
