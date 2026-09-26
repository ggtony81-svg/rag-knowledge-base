# -*- coding: utf-8 -*-
"""
端到端问答质量评测

前面的检索实验只测了"能不能找到正确文档"，本脚本测的是整个系统
（检索 + 大模型生成）最终给出的回答对不对。

数据：
    CMRC2018 验证集。每条样本是（问题, 原文段落, 标准答案片段），
    标准答案就是原文中标注的一段文字，属于"抽取式"答案。

评测指标：
    EM      预测答案与标准答案完全一致的比例
    F1      字符级 F1（中文按字切分，与 CMRC2018 官方评测口径一致）
    包含率  标准答案是否出现在回答中

    为什么三个都要报：CMRC2018 的标准答案是原文里的一小段（比如"5天"），
    而大模型倾向于组织成完整句子作答（"根据手册，入职满一年可享受 5 天带薪年假"）。
    这种情况下包含率能命中，但 EM 会判错、F1 也会被多余的字拉低。
    只看 EM 会低估系统实际可用性，只报包含率又显得在挑好看的指标，
    所以三个一起报，并在论文里说明各自的口径差别。

失败归因：
    对每条样本额外记录"检索到的前 k 段里有没有标准答案所在的那一段"。
    这样可以把答错的样本分成两类：
        检索没找到   —— 问题出在检索环节
        找到了却没答对 —— 问题出在生成环节
    两类问题的改进方向完全不同，分开统计才有意义。

用法：
    python eval/run_e2e_eval.py --limit 200
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataset
from retrievers import BM25Retriever

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")

# 与系统实际使用的一致：把检索到的前 3 段拼起来作为资料
DEFAULT_TOPK = 3

# 中文规范化的去标点表。评测前统一去掉标点和空白，
# 避免"5天"和"5 天。"被判成不一致。
_PUNCT = set("，。！？、；：""''（）《》【】…—·,.!?;:\"'()<>[]{}—-·~ \t\n\r")


def normalize(text):
    """评测前的规范化：转小写、去掉所有标点与空白。"""
    return "".join(ch for ch in str(text).lower() if ch not in _PUNCT)


def char_f1(pred, gold):
    """字符级 F1（中文按单字切分，统计口径同 CMRC2018 官方评测）。"""
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)


def score_one(pred, golds):
    """对一条预测打分，返回 (EM, F1, 是否包含)。多个标准答案取最优。"""
    if not pred:
        return 0.0, 0.0, False
    p_norm = normalize(pred)
    em, f1, contains = 0.0, 0.0, False
    for g in golds:
        g_norm = normalize(g)
        if not g_norm:
            continue
        if p_norm == g_norm:
            em = 1.0
        f1 = max(f1, char_f1(pred, g))
        if g_norm in p_norm:
            contains = True
    return em, f1, contains


# 两种系统提示词，用于对比"回答风格"对指标的影响
PROMPT_STYLES = {
    # 与系统实际使用的一致，测的是系统的真实行为
    "system": "基于以下资料回答问题：\n{ctx}",
    # 评测专用：只给答案本身。用于量化"回答风格"对 EM/F1 的影响，
    # 即 EM 偏低到底是答错了，还是答案被包在了完整句子里。
    "concise": ("基于以下资料回答问题。只给出答案本身，不要重复问题、"
                "不要解释、不要额外说明。\n{ctx}"),
}


def ask_deepseek(question, context, api_key, style="system",
                 model="deepseek-chat", timeout=60):
    """
    调用大模型生成回答。

    默认使用与 PDF 问答系统实际一致的提示词，保证评测的是系统的真实行为，
    而不是为了刷指标另换一套提示词。style="concise" 仅作为对照实验使用。
    """
    import requests
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={"model": model,
              "messages": [{"role": "system",
                            "content": PROMPT_STYLES[style].format(ctx=context)},
                           {"role": "user", "content": question}],
              "stream": False, "temperature": 0.7, "max_tokens": 512},
        timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200, help="抽样条数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    ap.add_argument("--sleep", type=float, default=0.0, help="每次调用后的间隔秒数")
    ap.add_argument("--style", default="system", choices=["system", "concise"],
                    help="提示词风格：system=系统实际使用；concise=只给答案（对照实验）")
    args = ap.parse_args()

    from config import DEEPSEEK_API_KEY

    print("=" * 78)
    print("端到端问答质量评测 — 数据集：CMRC2018 验证集")
    print("=" * 78)

    qa = dataset.load_e2e_qa()
    print(f"验证集规模：{len(qa)} 条（含标准答案）")

    rng = random.Random(args.seed)
    idx = list(range(len(qa)))
    rng.shuffle(idx)
    idx = idx[:args.limit]
    sample = qa.iloc[idx].reset_index(drop=True)
    print(f"本次抽样  ：{len(sample)} 条（随机种子 {args.seed}）")
    print()

    # 检索语料 = 验证集里出现过的所有原文段落（去重）
    contexts = sorted(qa["context"].unique().tolist())
    ctx_ids = [f"c{i}" for i in range(len(contexts))]
    retriever = BM25Retriever()
    t0 = time.time()
    retriever.index(ctx_ids, contexts)
    print(f"检索语料  ：{len(contexts)} 篇（建立索引 {time.time() - t0:.1f} 秒）")
    print(f"检索方案  ：BM25 关键词检索（单阶段方案里 P@1 最高且最快）")
    print(f"生成模型  ：deepseek-chat，资料取检索前 {args.topk} 段")
    print(f"提示词风格：{args.style}"
          + ("（与系统实际使用一致）" if args.style == "system"
             else "（对照实验：只给答案本身）"))
    print()
    print("开始评测（每条需要调用一次大模型接口）...")

    records = []
    n_fail = 0
    t_start = time.time()

    for i, row in sample.iterrows():
        q = row["question"]
        golds = list(row["answers"])
        gold_ctx = row["context"]

        hits = retriever.search(q, topk=args.topk)
        got_ids = [d for d, _ in hits]
        ctx = "\n".join(contexts[int(cid[1:])] for cid in got_ids)

        # 检索是否命中标准答案所在的段落
        retrieval_hit = gold_ctx in [contexts[int(cid[1:])] for cid in got_ids]

        try:
            pred = ask_deepseek(q, ctx, DEEPSEEK_API_KEY, style=args.style)
        except Exception as e:
            pred = ""
            n_fail += 1
            print(f"  [{i + 1}/{len(sample)}] 调用失败：{e}")

        em, f1, contains = score_one(pred, golds)
        records.append({
            "question": q,
            "golds": golds,
            "pred": pred,
            "EM": em, "F1": f1, "contains": contains,
            "retrieval_hit": retrieval_hit,
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(sample):
            done = records
            print(f"  [{i + 1}/{len(sample)}]  "
                  f"平均 EM={sum(r['EM'] for r in done) / len(done):.4f}  "
                  f"F1={sum(r['F1'] for r in done) / len(done):.4f}  "
                  f"包含率={sum(r['contains'] for r in done) / len(done):.4f}")

        if args.sleep:
            time.sleep(args.sleep)

    # ---------------- 汇总 ----------------
    n = len(records)
    em = sum(r["EM"] for r in records) / n
    f1 = sum(r["F1"] for r in records) / n
    contains = sum(r["contains"] for r in records) / n

    hit = [r for r in records if r["retrieval_hit"]]
    miss = [r for r in records if not r["retrieval_hit"]]
    hit_ok = sum(r["contains"] for r in hit) / len(hit) if hit else 0.0
    miss_ok = sum(r["contains"] for r in miss) / len(miss) if miss else 0.0

    print()
    print("=" * 78)
    print("汇总")
    print("=" * 78)
    print(f"样本数        : {n}" + (f"（接口调用失败 {n_fail} 条，按答错计入）" if n_fail else ""))
    print(f"EM            : {em:.4f}")
    print(f"F1            : {f1:.4f}")
    print(f"包含率        : {contains:.4f}")
    print()
    print("失败归因")
    print(f"  检索命中标准段落 : {len(hit)}/{n} ({len(hit) / n * 100:.1f}%)，其中答对 {hit_ok * 100:.1f}%")
    print(f"  检索未命中       : {len(miss)}/{n} ({len(miss) / n * 100:.1f}%)，其中答对 {miss_ok * 100:.1f}%")
    print()
    print(f"总耗时 {time.time() - t_start:.0f} 秒")

    os.makedirs(RESULT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULT_DIR, f"e2e_{stamp}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"limit": args.limit, "seed": args.seed, "topk": args.topk,
                       "retriever": "BM25", "model": "deepseek-chat",
                       "prompt_style": args.style},
            "summary": {"n": n, "n_fail": n_fail, "EM": em, "F1": f1,
                        "contains": contains,
                        "retrieval_hit_rate": len(hit) / n,
                        "acc_when_hit": hit_ok,
                        "acc_when_miss": miss_ok},
            "records": records,
        }, f, ensure_ascii=False, indent=2)
    print(f"结果已保存：{out}")


if __name__ == "__main__":
    main()
