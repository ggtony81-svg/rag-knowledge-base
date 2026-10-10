# -*- coding: utf-8 -*-
"""
超参数调节实验

指导书 5.5 节要求"记录多组超参数调节实验的结果，分析学习率、批次大小等对模型
性能的影响，确定最优超参数组合"。本脚本就是干这件事的。

设计上的几个取舍，都得说清楚：

1. **用训练集的子集**。全量 8031 句对跑 3 轮要 3.6 小时（CPU），扫 5 组就是 18 小时。
   调参用子集、最终模型用全量，是常规做法，不是偷工减料——但论文里必须写明，
   否则读者会以为表里的模型和线上模型是同一个训练规模。

2. **所有配置用同一个子集**。子集按固定种子打乱后取前 N 条，每组配置拿到的
   训练数据完全一样。不然"学习率的影响"里混进了"训练数据的影响"，结论就没意义。

3. **固定轮数，不用早停**。早停会让不同配置跑的总步数不同，没法比较。
   所以耐心值设得很大，每组都跑满同样的轮数。

4. **扫完直接用检索指标评估**，而不是只看验证损失。因为最终要的是检索 P@1，
   验证损失低的配置不一定检索就好——这一点在 4.5 节的分块实验里已经见过一次了。

用法：
    python eval/tune_hyperparams.py                  # 跑全部配置
    python eval/tune_hyperparams.py --only lr1e-5    # 只跑指定配置（断点续跑用）
    python eval/tune_hyperparams.py --list           # 只列出配置
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULT_DIR = os.path.join(BASE_DIR, "eval", "results")
TUNE_CACHE = os.path.join(BASE_DIR, "eval", "cache", "tune")
OUT_JSON = os.path.join(RESULT_DIR, "hparam_sweep.json")

# 基线就是线上实际用的那组：lr=2e-5, batch=16, epochs=3。
# 这里用 2 轮而不是 3 轮，是为了控制总耗时（5 组 × 3 轮 ≈ 3 小时，太久）；
# 轮数对所有配置一致，所以组间比较仍然成立。
EPOCHS = 2
TRAIN_SUBSET = 1000
# --smoke 会把上面两个值临时改小，几分钟跑通整条链路（训练→保存→评测）。
# 目的是在启动 2 小时的长跑之前先确认脚本本身没问题。
SMOKE_EPOCHS = 1
SMOKE_SUBSET = 32
MAX_SEQ_LEN = 512
WARMUP_RATIO = 0.1
SEED = 42

CONFIGS = [
    {"tag": "lr1e-5", "lr": 1e-5, "batch_size": 16, "changed": "学习率 1e-5"},
    {"tag": "lr2e-5", "lr": 2e-5, "batch_size": 16, "changed": "基线（学习率 2e-5，批大小 16）"},
    {"tag": "lr5e-5", "lr": 5e-5, "batch_size": 16, "changed": "学习率 5e-5"},
    {"tag": "bs8", "lr": 2e-5, "batch_size": 8, "changed": "批大小 8"},
    {"tag": "bs32", "lr": 2e-5, "batch_size": 32, "changed": "批大小 32"},
]

METRIC_RE = re.compile(
    r"P@1=([\d.]+)\s+Recall@1=([\d.]+)\s+Recall@3=([\d.]+)\s+Recall@5=([\d.]+)\s+"
    r"Recall@10=([\d.]+)\s+MRR@10=([\d.]+)\s+nDCG@10=([\d.]+)")


def load_sweep():
    """读回已有结果，支持中断后续跑。"""
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {"settings": {"epochs": EPOCHS, "train_subset": TRAIN_SUBSET,
                         "max_seq_length": MAX_SEQ_LEN, "warmup_ratio": WARMUP_RATIO,
                         "seed": SEED,
                         "note": "调参用训练集子集；最终模型仍用全量数据训练，"
                                 "两者的训练规模不同，论文中需注明。"},
            "runs": []}


def save_sweep(data):
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_subset(train, model, batch_size, subset_size):
    """
    按固定种子取训练子集。

    先打乱再取前 N——直接取前 N 会让子集里的文档种类偏少（数据集是
    按文档顺序排的），不同配置虽然拿到同样的数据，但代表性和随机取不一样。
    """
    import random as _random
    idx = list(range(len(train)))
    _random.Random(SEED).shuffle(idx)
    idx = idx[:subset_size]
    sub = train.iloc[idx].reset_index(drop=True)
    return [(r.question, r.context) for r in sub.itertuples()]


def eval_retrieval(model_dir):
    """用微调后的模型跑一遍检索评测，返回各项指标。"""
    cmd = [sys.executable, os.path.join(BASE_DIR, "eval", "run_retrieval_eval.py"),
           "--schemes", "dense_chunk_ft", "--ft-model", model_dir]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_ENDPOINT="https://hf-mirror.com")
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, cwd=BASE_DIR)
    m = METRIC_RE.search(p.stdout or "")
    if not m:
        return None, (p.stdout or "")[-800:] + "\n--- stderr ---\n" + (p.stderr or "")[-800:]
    keys = ["P@1", "Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR@10", "nDCG@10"]
    return {k: float(v) for k, v in zip(keys, m.groups())}, None


def run_one(cfg, smoke=False, threads=None):
    """训练一组配置，然后评测。返回结果字典。"""
    os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 8))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    epochs = SMOKE_EPOCHS if smoke else EPOCHS
    subset = SMOKE_SUBSET if smoke else TRAIN_SUBSET

    import torch
    from sentence_transformers import SentenceTransformer, losses
    from finetune_embedding import (load_training_data, make_batches, run_epoch,
                                    set_seed, EMBED_MODEL_NAME)

    n_threads = threads or max(1, (os.cpu_count() or 8) - 4)
    torch.set_num_threads(n_threads)
    print(f"  使用 {n_threads} 个线程（留出核心给正在运行的服务，截图不会太卡）")
    set_seed(SEED)

    out_dir = os.path.join(TUNE_CACHE, cfg["tag"])
    print("=" * 78)
    print(f"[{cfg['tag']}] {cfg['changed']}")
    print("=" * 78)

    train, val, stats = load_training_data(seed=SEED)
    model = SentenceTransformer(EMBED_MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LEN

    train_samples = make_subset(train, model, cfg["batch_size"], subset)
    val_samples = [(r.question, r.context) for r in val.itertuples()]

    train_batches = make_batches(train_samples, cfg["batch_size"], shuffle=True, seed=SEED)
    val_batches = make_batches(val_samples, cfg["batch_size"], shuffle=False)

    print(f"  训练子集 {len(train_samples)} 句对 -> {len(train_batches)} 批/轮，共 {epochs} 轮")
    print(f"  验证集   {len(val_samples)} 句对 -> {len(val_batches)} 批")
    print(f"  超参数   lr={cfg['lr']} batch={cfg['batch_size']} warmup={WARMUP_RATIO}")

    train_loss = losses.MultipleNegativesRankingLoss(model)
    total_steps = len(train_batches) * epochs
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup_steps) * max(0.0, 1 - step / max(1, total_steps)),
    )

    epochs_rec, best_val, best_epoch = [], float("inf"), -1
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loss, train_batches, optimizer, scheduler,
                       desc=f"epoch{epoch}")
        va = run_epoch(model, train_loss, val_batches)
        epochs_rec.append({"epoch": epoch, "train_loss": round(tr, 6),
                           "val_loss": round(va, 6),
                           "lr": scheduler.get_last_lr()[0]})
        print(f"  epoch {epoch}: train_loss={tr:.4f}  val_loss={va:.4f}")
        if va < best_val:
            best_val, best_epoch = va, epoch
            os.makedirs(out_dir, exist_ok=True)
            model.save(out_dir)

    train_sec = round(time.time() - t0, 1)
    print(f"  训练完成，最优 epoch={best_epoch} 验证损失={best_val:.4f}，耗时 {train_sec/60:.1f} 分钟")

    print("  正在评测检索效果 ...")
    metrics, err = eval_retrieval(out_dir)

    rec = {
        "tag": cfg["tag"],
        "changed": cfg["changed"],
        "lr": cfg["lr"],
        "batch_size": cfg["batch_size"],
        "epochs_run": epochs,
        "smoke": smoke,
        "train_pairs_used": len(train_samples),
        "train_batches_per_epoch": len(train_batches),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val, 6),
        "epochs": epochs_rec,
        "train_sec": train_sec,
        "model_dir": os.path.relpath(out_dir, BASE_DIR),
    }
    if metrics:
        rec["metrics"] = metrics
        print(f"  检索 P@1 = {metrics['P@1']:.4f}  nDCG@10 = {metrics['nDCG@10']:.4f}")
    else:
        rec["metrics_error"] = err
        print(f"  评测失败：{err[:300]}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="只跑这些 tag")
    ap.add_argument("--list", action="store_true", help="只列出配置")
    ap.add_argument("--threads", type=int, default=None,
                    help="训练线程数，默认留 4 个核给其它程序")
    ap.add_argument("--smoke", action="store_true",
                    help=f"冒烟测试：子集 {SMOKE_SUBSET} 条、{SMOKE_EPOCHS} 轮，几分钟跑通链路")
    args = ap.parse_args()

    if args.list:
        for c in CONFIGS:
            print(f"  {c['tag']:8s} lr={c['lr']:<7g} batch={c['batch_size']:<3d} {c['changed']}")
        return 0

    # 冒烟结果写到单独的文件：这些数是从 32 条样本上训出来的，
    # 混进正式结果里会被误当成实验数据。
    global OUT_JSON
    if args.smoke:
        OUT_JSON = os.path.join(RESULT_DIR, "hparam_sweep_smoke.json")

    data = load_sweep()
    done = {r["tag"] for r in data["runs"]}
    todo = [c for c in CONFIGS
            if c["tag"] not in done and (not args.only or c["tag"] in args.only)]

    if not todo:
        print("没有需要跑的配置。已完成：", sorted(done))
        return 0

    if args.smoke:
        print(f"*** 冒烟模式：子集 {SMOKE_SUBSET} 条 / {SMOKE_EPOCHS} 轮，"
              f"结果仅供验证脚本，不是实验数据 ***\n")
    else:
        est_min = len(todo) * 25
        print(f"待跑 {len(todo)} 组：{[c['tag'] for c in todo]}")
        print(f"预计耗时约 {est_min} 分钟（CPU，无 GPU）\n")

    for cfg in todo:
        try:
            rec = run_one(cfg, smoke=args.smoke, threads=args.threads)
        except Exception as e:
            rec = {"tag": cfg["tag"], "changed": cfg["changed"], "lr": cfg["lr"],
                   "batch_size": cfg["batch_size"], "error": f"{type(e).__name__}: {e}"}
            print(f"  [{cfg['tag']}] 出错：{rec['error']}")
        data["runs"].append(rec)
        save_sweep(data)          # 每组跑完立刻落盘，中断也不丢
        print(f"  已写入 {OUT_JSON}\n")

    print("=" * 78)
    print("汇总")
    print("=" * 78)
    print(f"{'配置':<10}{'lr':>8}{'batch':>7}{'最佳轮':>7}{'验证损失':>11}{'检索P@1':>10}{'nDCG@10':>10}")
    for r in data["runs"]:
        if "metrics" in r:
            m = r["metrics"]
            print(f"{r['tag']:<10}{r['lr']:>8g}{r['batch_size']:>7}{r['best_epoch']:>7}"
                  f"{r['best_val_loss']:>11.4f}{m['P@1']:>10.4f}{m['nDCG@10']:>10.4f}")
        else:
            print(f"{r['tag']:<10}{r['lr']:>8g}{r['batch_size']:>7}   评测未完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
