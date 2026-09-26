# -*- coding: utf-8 -*-
"""
微调向量检索模型

目的：
    实验发现通用向量模型 BAAI/bge-small-zh-v1.5 在本数据集上的检索效果
    低于 BM25 关键词检索（P@1 0.8590 对 0.9164）。原因是该模型在通用语料上
    预训练，没有见过本领域"问题—段落"的对应关系。

    本脚本用 CMRC2018 训练集的问题-段落对微调该模型，使其学会本领域的
    "哪条问题该匹配哪段文档"，再回到同一评测集上对比微调前后的效果。

做法：
    采用对比学习中的「批内负样本」策略（MultipleNegativesRankingLoss）。
    一个批次里有 B 个 (问题, 正例段落) 对，对每条问题而言，批次内其余 B-1
    个段落自动充当负样本——模型要把自己的正例排到最前，就必须把问题与
    正确段落的向量拉近、与错误段落推远。这样不需要人工标注负样本。

    损失函数（InfoNCE 形式）：
        L = -1/B · Σ_i log( exp(sim(qi, di+) / τ) / Σ_j exp(sim(qi, dj) / τ) )
    其中 sim 为余弦相似度，τ 为温度系数。

    因为负样本来自同一批次内部，批次越大负样本越多、训练信号越强，
    所以 batch size 对效果影响明显（本项目取 16，受 CPU 算力限制）。

数据处理：
    训练前会剔除与评测语料重叠的文档（实测 204 篇，占评测语料 21.8%），
    否则模型在训练中已见过评测文档，评测结果会虚高。

用法：
    python eval/finetune_embedding.py                    # 完整训练
    python eval/finetune_embedding.py --max-steps 30     # 只跑少量步数（调试）
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset
from retrievers import EMBED_MODEL_NAME

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
RESULT_DIR = os.path.join(BASE_DIR, "results")
DEFAULT_OUT = os.path.join(CACHE_DIR, "finetuned_model")


def set_seed(seed=42):
    """固定随机种子，保证训练过程可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)


def load_training_data(seed=42, val_ratio=0.05):
    """
    加载并清洗训练数据，按文档划分训练集与验证集。

    划分以"文档"为单位而不是以"句子对"为单位——同一篇文档可能对应多个问题，
    若按句子对随机划分，同一篇文档会同时出现在训练集和验证集里，
    验证损失就失去意义了。
    """
    corpus, _, _ = dataset.build_retrieval_benchmark(dedup=True)
    eval_contexts = set(corpus["text"].tolist())

    pairs = dataset.load_sft_pairs(exclude_contexts=eval_contexts)

    contexts = sorted(pairs["context"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(contexts)
    n_val = max(1, int(len(contexts) * val_ratio))
    val_contexts = set(contexts[:n_val])
    train_contexts = set(contexts[n_val:])

    train = pairs[pairs["context"].isin(train_contexts)].reset_index(drop=True)
    val = pairs[pairs["context"].isin(val_contexts)].reset_index(drop=True)

    stats = {
        "eval_corpus_size": len(eval_contexts),
        "dropped_leak_pairs": pairs.attrs.get("n_dropped_leak", 0),
        "train_pairs": len(train),
        "train_contexts": len(train_contexts),
        "val_pairs": len(val),
        "val_contexts": len(val_contexts),
    }
    return train, val, stats


def make_batches(samples, batch_size, shuffle, seed=42):
    """把 (问题, 段落) 样本切成批次。批次内每个问题都有对应的正例段落，构成对角线。"""
    idx = list(range(len(samples)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    batches = []
    for i in range(0, len(idx) - batch_size + 1, batch_size):   # 丢弃末尾不足一批的样本
        batches.append([samples[j] for j in idx[i:i + batch_size]])
    return batches


def encode_features(model, texts):
    """把文本转成模型需要的输入张量。新版 sentence-transformers 用 preprocess 代替 tokenize。"""
    fn = getattr(model, "preprocess", None) or model.tokenize
    return fn(list(texts))


def run_epoch(model, loss_fn, batches, optimizer=None, scheduler=None, desc="", log=None):
    """
    跑一个 epoch。传 optimizer 表示训练，不传表示只做验证（不算梯度）。
    返回平均损失。
    """
    import torch

    training = optimizer is not None
    model.train() if training else model.eval()

    total, n = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in batches:
            qs = [s[0] for s in batch]
            ds = [s[1] for s in batch]
            fq = encode_features(model, qs)
            fd = encode_features(model, ds)
            # 批内负样本：第 i 条问题的正例就是第 i 个段落，构成对角线标签
            labels = torch.arange(len(qs), dtype=torch.long)

            if training:
                optimizer.zero_grad()
            loss = loss_fn([fq, fd], labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total += float(loss.detach())
            n += 1
            if log is not None:
                log.append(round(float(loss.detach()), 6))
            if training and n % 50 == 0:
                print(f"    {desc} 批次 {n}  loss={total / n:.4f}"
                      + (f"  lr={scheduler.get_last_lr()[0]:.2e}" if scheduler else ""))
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=EMBED_MODEL_NAME, help="基础模型")
    ap.add_argument("--out", default=DEFAULT_OUT, help="微调后模型保存目录")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--patience", type=int, default=2, help="早停耐心值（按验证损失）")
    ap.add_argument("--max-steps", type=int, default=None, help="限制总步数（调试用）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 8))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    import torch
    from sentence_transformers import SentenceTransformer, losses

    torch.set_num_threads(os.cpu_count() or 8)
    set_seed(args.seed)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("=" * 78)
    print("微调向量检索模型")
    print("=" * 78)

    train, val, stats = load_training_data(seed=args.seed)
    print(f"基础模型        : {args.model}")
    print(f"剔除泄漏句对    : {stats['dropped_leak_pairs']} 条（训练语料与评测语料重叠）")
    print(f"训练集          : {stats['train_pairs']} 句对 / {stats['train_contexts']} 篇文档")
    print(f"验证集          : {stats['val_pairs']} 句对 / {stats['val_contexts']} 篇文档")
    print(f"超参数          : epochs={args.epochs} batch={args.batch_size} lr={args.lr} "
          f"max_len={args.max_seq_length} warmup={args.warmup_ratio} patience={args.patience}")
    print()

    model = SentenceTransformer(args.model)
    model.max_seq_length = args.max_seq_length

    train_samples = [(r.question, r.context) for r in train.itertuples()]
    val_samples = [(r.question, r.context) for r in val.itertuples()]

    train_batches = make_batches(train_samples, args.batch_size, shuffle=True, seed=args.seed)
    val_batches = make_batches(val_samples, args.batch_size, shuffle=False)
    if args.max_steps:
        train_batches = train_batches[:args.max_steps]

    train_loss = losses.MultipleNegativesRankingLoss(model)

    total_steps = len(train_batches) * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup_steps) * max(0.0, 1 - step / max(1, total_steps)),
    )

    history = {
        "config": vars(args),
        "stats": stats,
        "epochs": [],
        "batch_loss_epoch1": [],
        "batches_per_epoch": len(train_batches),
        "total_epoch_steps": total_steps,
    }
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    global_step = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        print(f"epoch {epoch}/{args.epochs} 训练中（共 {len(train_batches)} 批）...")

        batch_log = history["batch_loss_epoch1"] if epoch == 1 else None
        train_avg = run_epoch(model, train_loss, train_batches, optimizer, scheduler,
                              desc=f"epoch{epoch}", log=batch_log)
        global_step += len(train_batches)

        print(f"epoch {epoch} 验证中（共 {len(val_batches)} 批）...")
        val_avg = run_epoch(model, train_loss, val_batches)

        ep = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": round(train_avg, 6),
            "val_loss": round(val_avg, 6),
            "lr": scheduler.get_last_lr()[0],
            "elapsed_sec": round(time.time() - t_epoch, 1),
        }
        history["epochs"].append(ep)
        print(f"epoch {epoch} 完成  train_loss={train_avg:.4f}  val_loss={val_avg:.4f}  "
              f"耗时 {time.time() - t_epoch:.0f}s")

        if val_avg < best_val:
            best_val, best_epoch = val_avg, epoch
            bad_epochs = 0
            os.makedirs(args.out, exist_ok=True)
            model.save(args.out)
            print(f"  验证损失创新低，已保存模型 -> {args.out}")
        else:
            bad_epochs += 1
            print(f"  验证损失未下降（{bad_epochs}/{args.patience}）")
            if bad_epochs >= args.patience:
                print("  触发早停，结束训练")
                break

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = round(best_val, 6)
    history["total_time_sec"] = round(time.time() - t_start, 1)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    hist_path = os.path.join(RESULT_DIR, f"finetune_history_{stamp}.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print()
    print(f"训练结束。最优 epoch={best_epoch}，最优验证损失={best_val:.4f}")
    print(f"总耗时 {history['total_time_sec'] / 60:.1f} 分钟")
    print(f"训练曲线数据：{hist_path}")
    print(f"模型保存于：{args.out}")
    print()
    print("下一步：用微调后的模型重新评测检索效果")
    print(f"  python eval/run_retrieval_eval.py --schemes dense_ft --ft-model {args.out}")


if __name__ == "__main__":
    main()
