# -*- coding: utf-8 -*-
"""
检索评价指标

本项目评测的检索任务定义：给定一条查询，从语料库中找出与它相关的那篇文档。
标准答案由 qrels 给出，且每条查询只对应 1 篇相关文档。

在这个设定下使用的指标：

  Recall@k  前 k 条结果里是否命中了那篇相关文档（命中记 1，否则记 0），
            对所有查询取平均。因每条查询只有 1 篇相关文档，该值等价于命中率
            （Hit Rate@k）。

  MRR@k     第一篇相关文档排名倒数（1/rank）的平均值。只关心"第一次命中排多前"，
            排名越靠前越接近 1，一次都没命中记 0。

  nDCG@k    归一化折损累计增益。因为每条查询只有 1 篇相关文档，
            理想排序的 DCG 为 1/log2(1+1)=1，因此退化为 1/log2(rank+1)，
            与 MRR 的区别是它对靠前的位置给更高的权重。

  P@1       排名第一的位置命中的比例，可看作"最坏情况下用户只看第一条"的效果。
"""
import math


def recall_at_k(ranked_ids, relevant_ids, k):
    """前 k 条中是否包含相关文档。"""
    return 1.0 if any(d in relevant_ids for d in ranked_ids[:k]) else 0.0


def mrr_at_k(ranked_ids, relevant_ids, k):
    """第一篇相关文档的排名倒数。"""
    for rank, d in enumerate(ranked_ids[:k], start=1):
        if d in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids, relevant_ids, k):
    """归一化折损累计增益（每查询单相关文档时退化为 1/log2(rank+1)）。"""
    for rank, d in enumerate(ranked_ids[:k], start=1):
        if d in relevant_ids:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def precision_at_1(ranked_ids, relevant_ids):
    """Top-1 命中率。"""
    return 1.0 if ranked_ids and ranked_ids[0] in relevant_ids else 0.0


def evaluate(all_ranked, qrels, ks=(1, 3, 5, 10), mrr_k=10):
    """
    对全部查询统一算指标。

    参数
        all_ranked : {qid: [doc_id, ...]}  每条查询的检索结果（按分数降序）
        qrels      : {qid: set(doc_id)}    标准答案
        ks         : 要统计的 k 值
        mrr_k      : MRR 截断位置

    返回
        {指标名: 数值}，同时附带命中的查询数/总数，便于核对。
    """
    qids = [q for q in all_ranked if q in qrels]
    n = len(qids)
    if n == 0:
        return {}

    out = {}
    for k in ks:
        out[f"Recall@{k}"] = sum(recall_at_k(all_ranked[q], qrels[q], k) for q in qids) / n
    out[f"MRR@{mrr_k}"] = sum(mrr_at_k(all_ranked[q], qrels[q], mrr_k) for q in qids) / n
    out["nDCG@10"] = sum(ndcg_at_k(all_ranked[q], qrels[q], 10) for q in qids) / n
    out["P@1"] = sum(precision_at_1(all_ranked[q], qrels[q]) for q in qids) / n
    out["_queries"] = n
    return out


def format_result(name, res, extra=None):
    """把一次评测结果格式化成一行表格文本。"""
    if not res:
        return f"{name:<20} 无结果"
    parts = [f"{name:<20}"]
    for key in ["P@1", "Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR@10", "nDCG@10"]:
        if key in res:
            parts.append(f"{key}={res[key]:.4f}")
    line = "  ".join(parts)
    if extra:
        line += "  " + "  ".join(f"{k}={v}" for k, v in extra.items())
    return line
