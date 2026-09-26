# -*- coding: utf-8 -*-
"""
数据集加载与清洗模块

数据来源（均为公开数据集，已下载到 eval/data/）：
  1. hfl/cmrc2018                                —— 微调训练集 + 端到端评测集
  2. jinaai/longcontext-cmrc2018-zh              —— 检索评测的语料库与查询
  3. jinaai/longcontext-cmrc2018-zh-qrels        —— 检索评测的标准答案（qrels）

本模块只做"读数据 + 清洗 + 统计"，不含任何检索或模型逻辑，
保证数据集层面的处理过程可复现、可检查。
"""
import os
import json
import hashlib
from collections import Counter

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 文件名常量，避免各处重复拼字符串
F_TRAIN = "cmrc_train.parquet"
F_VAL = "cmrc_val.parquet"
F_CORPUS = "lc_corpus.parquet"
F_QUERIES = "lc_queries.parquet"
F_QRELS = "lc_qrels.parquet"


def _read(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到数据文件 {path}。请先运行 eval/download_data.py 下载数据集。"
        )
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# 一、检索评测数据（知识库语料 + 查询 + 标准答案）
# ---------------------------------------------------------------------------

def load_corpus(dedup=True):
    """
    加载知识库语料。

    dedup=True 时按文本内容去重：原始语料 1149 条里有 215 条是完全相同的文本，
    重复文档会让同一篇内容在检索结果里占多个位置，虚高召回率，因此需要去重。
    去重时保留首次出现的 id，并把被合并掉的 id 记录在返回值的 attrs 中，
    这样 qrels 里的 pid 仍然能被映射回去重后的文档。
    """
    df = _read(F_CORPUS)
    df = df.dropna(subset=["id", "text"]).reset_index(drop=True)

    n_raw = len(df)
    id_map = {}          # 原始 id -> 保留的 id
    if dedup:
        kept_rows = []
        seen = {}        # text 的 md5 -> 保留的 id
        for _, row in df.iterrows():
            key = hashlib.md5(row["text"].encode("utf-8")).hexdigest()
            if key in seen:
                id_map[row["id"]] = seen[key]
            else:
                seen[key] = row["id"]
                id_map[row["id"]] = row["id"]
                kept_rows.append({"id": row["id"], "text": row["text"]})
        df = pd.DataFrame(kept_rows)
    else:
        id_map = {i: i for i in df["id"]}

    df = df.reset_index(drop=True)
    df.attrs["n_raw"] = n_raw
    df.attrs["n_dedup"] = len(df)
    df.attrs["id_map"] = id_map
    return df


def load_queries():
    """加载检索评测的查询（1149 条）。"""
    df = _read(F_QUERIES).dropna(subset=["id", "text"]).reset_index(drop=True)
    return df


def load_qrels(id_map=None):
    """
    加载检索标准答案。

    原始 qrels 是 (qid, pid, score) 三元组，每条查询只有 1 篇相关文档（score 均为 1）。
    id_map 用于把 pid 映射到去重后的语料 id（corpus 去重后必须传）。
    """
    df = _read(F_QRELS).dropna(subset=["qid", "pid"]).reset_index(drop=True)
    df["score"] = df["score"].astype(int)
    if id_map is not None:
        df["pid"] = df["pid"].map(lambda p: id_map.get(p, p))
    return df


def build_retrieval_benchmark(dedup=True):
    """
    组装完整的检索评测集，返回 (corpus, queries, qrels)。

    只保留"查询有标准答案、且标准答案文档确实存在于语料库中"的记录，
    避免评测时出现无法判定的查询。
    """
    corpus = load_corpus(dedup=dedup)
    queries = load_queries()
    qrels = load_qrels(id_map=corpus.attrs["id_map"])

    valid_doc_ids = set(corpus["id"])
    qrels = qrels[qrels["pid"].isin(valid_doc_ids)].reset_index(drop=True)
    valid_qids = set(qrels["qid"])
    queries = queries[queries["id"].isin(valid_qids)].reset_index(drop=True)

    # 每条查询至少要有 1 篇相关文档
    per_query = Counter(qrels["qid"])
    queries = queries[queries["id"].map(lambda q: per_query.get(q, 0) > 0)].reset_index(drop=True)
    qrels = qrels[qrels["qid"].isin(set(queries["id"]))].reset_index(drop=True)

    return corpus, queries, qrels


# ---------------------------------------------------------------------------
# 二、微调训练数据（CMRC2018 官方训练集）
# ---------------------------------------------------------------------------

def load_sft_pairs(limit=None, min_len=10, dedup=True, exclude_contexts=None):
    """
    从 CMRC2018 训练集构造 (问题, 正例文档) 句对，用于微调向量模型。

    CMRC2018 的每条样本是 (context, question, answers)，其中 answers 的 answer_start
    指明了答案在 context 中的位置。用 context 整体作为正例文档即可——模型要学的正是
    "这条问题该匹配哪段文档"。

    清洗动作：
      - 丢掉 answers 为空或 answer_start 越界的脏样本
      - 丢掉 context 或 question 过短的样本
      - 同一 (question, context) 组合去重
      - 剔除 context 出现在 exclude_contexts 中的样本（见下方说明）

    关于 exclude_contexts：
      实测发现 CMRC2018 训练集与本项目检索评测所用的语料库存在文本重叠——
      评测语料 934 篇中有 204 篇（21.8%）能在训练集里找到完全相同的原文。
      若不去除，微调时模型已经"见过"这些文档，评测分数会虚高，
      微调前后的对比也就不成立。因此训练前必须把评测语料中的文档从训练集剔除。
    """
    df = _read(F_TRAIN)

    rows = []
    n_bad_answer = 0
    n_leak = 0
    for _, r in df.iterrows():
        q = str(r["question"]).strip()
        c = str(r["context"]).strip()
        ans = r["answers"]
        starts = list(ans["answer_start"]) if isinstance(ans, dict) else []
        if not starts:
            n_bad_answer += 1
            continue
        if starts[0] < 0 or starts[0] > len(c):
            n_bad_answer += 1
            continue
        if len(q) < min_len or len(c) < min_len:
            continue
        if exclude_contexts is not None and c in exclude_contexts:
            n_leak += 1
            continue
        rows.append({"question": q, "context": c})

    pairs = pd.DataFrame(rows)
    n_before = len(pairs)
    if dedup:
        pairs = pairs.drop_duplicates(subset=["question", "context"]).reset_index(drop=True)

    if limit is not None:
        pairs = pairs.head(limit).reset_index(drop=True)

    pairs.attrs["n_raw"] = len(df)
    pairs.attrs["n_dropped_bad_answer"] = n_bad_answer
    pairs.attrs["n_dropped_leak"] = n_leak
    pairs.attrs["n_dedup"] = n_before - len(pairs)
    return pairs


# ---------------------------------------------------------------------------
# 三、端到端评测数据（CMRC2018 验证集，带标准答案）
# ---------------------------------------------------------------------------

def load_e2e_qa(limit=None):
    """
    加载端到端问答评测集：问题 + 标准答案 + 答案所在文档。

    返回的 DataFrame 每行:
        id, question, context, answers(标准答案列表)
    用于计算生成的回答与标准答案之间的 EM / F1。
    """
    df = _read(F_VAL)
    rows = []
    for _, r in df.iterrows():
        ans = r["answers"]
        texts = list(ans["text"]) if isinstance(ans, dict) else []
        texts = [t for t in texts if str(t).strip()]
        if not texts:
            continue
        rows.append({
            "id": r["id"],
            "question": str(r["question"]).strip(),
            "context": str(r["context"]).strip(),
            "answers": texts,
        })
    out = pd.DataFrame(rows)
    if limit is not None:
        out = out.head(limit).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# 四、数据统计（对应论文"数据探索性分析"一节）
# ---------------------------------------------------------------------------

def describe():
    """输出各数据集的基本统计信息，供 EDA 章节直接引用。"""
    info = {}

    corpus_raw = load_corpus(dedup=False)
    corpus = load_corpus(dedup=True)
    queries = load_queries()
    qrels = load_qrels()

    info["corpus_raw_docs"] = len(corpus_raw)
    info["corpus_unique_docs"] = len(corpus)
    info["corpus_dup_docs"] = len(corpus_raw) - len(corpus)
    info["corpus_len_mean"] = int(corpus["text"].str.len().mean())
    info["corpus_len_median"] = int(corpus["text"].str.len().median())
    info["corpus_len_min"] = int(corpus["text"].str.len().min())
    info["corpus_len_max"] = int(corpus["text"].str.len().max())

    info["query_count"] = len(queries)
    info["query_len_mean"] = int(queries["text"].str.len().mean())

    per_query = Counter(qrels["qid"])
    info["qrels_queries"] = len(per_query)
    info["qrels_per_query"] = dict(Counter(per_query.values()))

    train = _read(F_TRAIN)
    val = _read(F_VAL)
    info["train_pairs"] = len(train)
    info["train_contexts"] = int(train["context"].nunique())
    info["val_pairs"] = len(val)
    info["val_contexts"] = int(val["context"].nunique())
    info["train_val_context_overlap"] = len(set(train["context"]) & set(val["context"]))

    return info


if __name__ == "__main__":
    s = describe()
    print(json.dumps(s, ensure_ascii=False, indent=2))
