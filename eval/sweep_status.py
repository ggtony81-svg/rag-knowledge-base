# -*- coding: utf-8 -*-
"""
调参进度查询

跑步时候想知道"跑到哪了、还要多久"，不用去翻日志，
直接跑这个脚本就行：

    python eval/sweep_status.py

它读两个地方：
  eval/results/hparam_sweep.json  —— 已完成的配置（每组跑完立刻落盘）
  eval/results/hparam_sweep.log   —— 正在跑的那一组进行到哪了

注意预计时间只是个粗估：不同批大小的配置每轮批次数不一样
（bs8 是 bs16 的两倍，bs32 是一半），所以按已完成部分线性外推
只能给个大概范围，不是精确值。
"""
import json
import os
import re
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(BASE_DIR, "eval", "results")
JSON_PATH = os.path.join(RESULT_DIR, "hparam_sweep.json")
LOG_PATH = os.path.join(RESULT_DIR, "hparam_sweep.log")

# 与 tune_hyperparams.py 保持一致
TAGS = ["lr1e-5", "lr2e-5", "lr5e-5", "bs8", "bs32"]
BATCHES_PER_EPOCH = {"lr1e-5": 62, "lr2e-5": 62, "lr5e-5": 62, "bs8": 125, "bs32": 31}
EPOCHS = 2


def main():
    if not os.path.exists(LOG_PATH):
        print("找不到日志，调参似乎还没开始。")
        print(f"启动命令：python eval/tune_hyperparams.py --threads 8")
        return 1

    runs = []
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, encoding="utf-8") as f:
            runs = json.load(f).get("runs", [])

    done = {r["tag"] for r in runs if "metrics" in r}
    failed = {r["tag"] for r in runs if "metrics" not in r and "error" in r}

    # ---- 日志里正在跑哪一组、跑到第几轮 ----
    with open(LOG_PATH, encoding="utf-8", errors="replace") as f:
        log = f.read()

    current = None
    for tag in TAGS:
        if f"[{tag}]" in log:
            current = tag

    finished_epochs = len(re.findall(r"epoch \d+: train_loss=", log))

    print("=" * 62)
    print("调参进度")
    print("=" * 62)

    for tag in TAGS:
        if tag in done:
            r = next(x for x in runs if x["tag"] == tag)
            m = r["metrics"]
            print(f"  [完成] {tag:8s} P@1={m['P@1']:.4f}  nDCG@10={m['nDCG@10']:.4f}"
                  f"  用时 {r['train_sec']/60:.1f} 分钟")
        elif tag in failed:
            print(f"  [失败] {tag:8s} 见 {JSON_PATH}")
        elif tag == current:
            print(f"  [进行] {tag:8s} 训练中 ...")
        else:
            print(f"  [等待] {tag:8s}")

    print()

    if len(done) == len(TAGS):
        print("全部完成。结果在 eval/results/hparam_sweep.json")
        p1 = {r["tag"]: r["metrics"]["P@1"] for r in runs if "metrics" in r}
        best = max(p1, key=p1.get)
        print(f"检索 P@1 最高：{best}（{p1[best]:.4f}）")
        print()
        print("下一步：python eval/make_figures.py 出图，"
              "或把这张表写进论文 5.5 节")
        return 0

    # ---- 剩余时间粗估 ----
    t0 = os.path.getctime(LOG_PATH)
    elapsed = time.time() - t0
    total_epochs = sum(BATCHES_PER_EPOCH[t] for t in TAGS) * EPOCHS

    # 按"已完成批次"估算，比按 epoch 数准一些（不同组批次量差很多）
    done_batches = 0
    for tag in TAGS:
        if tag in done:
            done_batches += BATCHES_PER_EPOCH[tag] * EPOCHS
    if current and current not in done:
        done_batches += (finished_epochs % EPOCHS or EPOCHS) * BATCHES_PER_EPOCH[current] \
            if finished_epochs else 0
        done_batches = min(done_batches + BATCHES_PER_EPOCH[current] * (finished_epochs % EPOCHS or 0),
                           total_epochs)

    rate = elapsed / max(1, done_batches)          # 秒/批
    remain_batches = max(0, total_epochs - done_batches)
    remain_sec = remain_batches * rate + 3 * 60 * (len(TAGS) - len(done))   # 每组留 3 分钟评测

    print(f"  已完成 {len(done)}/{len(TAGS)} 组，"
          f"日志显示已跑完 {finished_epochs} 个 epoch")
    print(f"  已运行 {elapsed/60:.0f} 分钟，速度约 {rate:.1f} 秒/批")
    print(f"  预计还需约 {remain_sec/60:.0f} 分钟 "
          f"（约 {time.strftime('%H:%M', time.localtime(time.time() + remain_sec))} 完成）")
    print()
    print("  说明：这只是线性外推。bs8 那组批次量是 bs16 的两倍，会明显慢一截。")
    print(f"  原始日志：{LOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
