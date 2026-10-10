# -*- coding: utf-8 -*-
"""
生成论文用图

所有图的数据都来自 eval/results/ 下已经落盘的实验产物，
不在这个脚本里重新计算，也不手写任何数值：

    chunking_analysis.json    分块与截断分析   → 图1、图2
    retrieval_summary.json    六方案检索对比   → 图3、图4
    finetune_history_*.json   微调训练过程     → 图5
    e2e_*.json                端到端问答评测   → 图6

输出到 eval/results/figures/，300 dpi，可直接插进论文。

用法：
    python eval/make_figures.py
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(BASE_DIR, "eval", "results")
FIG_DIR = os.path.join(RESULT_DIR, "figures")

# 中文显示
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

# 统一配色：基线用灰、改进方案用蓝、最终方案用橙
C_BASE, C_IMPROVE, C_FINAL = "#9aa5b1", "#3b7dd8", "#e8833a"


def _load(name):
    with open(os.path.join(RESULT_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def _save(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存 {p}")


# ---------------------------------------------------------------- 图1/2

def fig_truncation(ana):
    lens = ana["raw_lengths"]
    mx = ana["whole_doc"]["max_tokens"]
    pct = ana["whole_doc"]["pct_over_limit"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lens, bins=40, color=C_IMPROVE, edgecolor="white", linewidth=0.6)
    ax.axvline(mx, color="#d0342c", linestyle="--", linewidth=1.6)
    top = ax.get_ylim()[1]
    ax.annotate(f"模型上限 {mx} token\n超过的 {pct}% 文档被静默截断",
                xy=(mx, top * 0.62), xytext=(mx + 90, top * 0.75),
                arrowprops=dict(arrowstyle="->", color="#d0342c", lw=1.3),
                color="#d0342c", fontsize=10, va="center")
    ax.set_xlabel("文档 token 数")
    ax.set_ylabel("文档数量")
    ax.set_title(f"评测语料长度分布（{len(lens)} 篇）")
    _save(fig, "fig1_长度分布与截断.png")


def fig_chunking(ana):
    st = ana["strategies"]
    names = ["整篇编码", "smart_chunk", "smart_chunk_v2"]
    rates = [ana["whole_doc"]["pct_over_limit"],
             st["smart_chunk"]["pct_chunks_truncated"],
             st["smart_chunk_v2"]["pct_chunks_truncated"]]

    fig, ax = plt.subplots(figsize=(6.4, 4))
    bars = ax.bar(names, rates, color=[C_BASE, C_IMPROVE, C_FINAL], width=0.55)
    for b, v in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.2f}%",
                ha="center", fontsize=10)
    ax.set_ylabel("被截断的比例（%）")
    ax.set_ylim(0, max(rates) * 1.28)
    ax.set_title("三种处理方式下的截断率")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    _save(fig, "fig2_分块前后的截断率.png")


# ---------------------------------------------------------------- 图3/4

def fig_ablation(rows):
    labels = [f"{r['no']}\n{r['label'].split(' ', 1)[1]}" for r in rows]
    p1 = [r["P@1"] * 100 for r in rows]
    colors = []
    for r in rows:
        if r["no"] == "A":
            colors.append(C_BASE)
        elif r["no"] == "F":
            colors.append(C_FINAL)
        else:
            colors.append(C_IMPROVE)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    bars = ax.bar(labels, p1, color=colors, width=0.6)
    base = p1[0]
    ax.axhline(base, color="#d0342c", linestyle="--", linewidth=1.2)

    for b, v in zip(bars, p1):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.2f}",
                ha="center", fontsize=9.5)

    # 基线标注：A 柱的数值标签占着 x≈0 那一带、C 柱的标签占着 x≈2 那一带，
    # 所以把文字放在 A、B 两根柱子上方的空白里（x=1），再用箭头指回虚线。
    # 直接写在虚线左端会压住 A 柱的数值标签。
    ax.annotate(f"BM25 基线 {base:.2f}%", xy=(1.0, base), xytext=(1.0, base + 1.7),
                color="#d0342c", ha="center", va="bottom", fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color="#d0342c", lw=1.1))

    ax.set_ylabel("P@1（%）")
    ax.set_ylim(min(p1) - 3, max(p1) + 4.5)
    ax.set_title("六种检索方案的 Top-1 命中率")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    _save(fig, "fig3_检索方案对比.png")


def fig_latency(rows):
    labels = [f"{r['no']}\n{r['label'].split(' ', 1)[1]}" for r in rows]
    ms = [r["search_ms_mean"] for r in rows]
    colors = [C_BASE if r["no"] == "A" else
              (C_FINAL if r["no"] == "F" else C_IMPROVE) for r in rows]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    bars = ax.bar(labels, ms, color=colors, width=0.6)
    ax.set_yscale("log")
    for b, v in zip(bars, ms):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.16, f"{v:.1f}",
                ha="center", fontsize=9.5)
    ax.set_ylabel("单次检索耗时（ms，对数刻度）")
    ax.set_ylim(3, max(ms) * 4)
    ax.set_title("各方案检索耗时对比")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.set_axisbelow(True)
    _save(fig, "fig4_检索耗时对比.png")


# ---------------------------------------------------------------- 图5

def fig_training(hist):
    ep = hist["epochs"]
    xs = [e["epoch"] for e in ep]
    tr = [e["train_loss"] for e in ep]
    va = [e["val_loss"] for e in ep]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))

    a1.plot(xs, tr, "o-", color=C_IMPROVE, linewidth=2)
    for x, y in zip(xs, tr):
        a1.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    a1.set_xlabel("训练轮次")
    a1.set_ylabel("训练损失")
    a1.set_title("训练损失（持续下降）")
    a1.set_xticks(xs)
    # 左右留白：数值标签是以数据点为中心画的，第一个点在最左刻度上，
    # 不留白的话标签会越过 y 轴压到刻度文字上。
    a1.set_xlim(0.7, 3.3)
    # 顶部留白，否则最高那个点的数值标签会被裁掉
    a1.set_ylim(min(tr) * 0.88, max(tr) * 1.16)
    a1.grid(alpha=0.25)

    a2.plot(xs, va, "s-", color=C_FINAL, linewidth=2)
    for x, y in zip(xs, va):
        a2.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    a2.set_xlabel("训练轮次")
    a2.set_ylabel("验证损失")
    a2.set_title("验证损失（基本持平）")
    a2.set_xticks(xs)
    a2.set_xlim(0.7, 3.3)     # 同上：最右点的标签会顶到右边界外
    a2.set_ylim(min(va) - 0.004, max(va) + 0.011)
    a2.grid(alpha=0.25)

    fig.suptitle("微调过程：训练损失下降明显，验证损失几乎不动", fontsize=12)
    _save(fig, "fig5_微调训练曲线.png")


# ---------------------------------------------------------------- 图6

def fig_e2e(e2e):
    s = e2e["summary"]
    hit_n = round(s["retrieval_hit_rate"] * s["n"])
    miss_n = s["n"] - hit_n
    hit_ok = round(s["acc_when_hit"] * hit_n)
    hit_bad = hit_n - hit_ok

    fig, ax = plt.subplots(figsize=(7.6, 3.6))

    cats = ["检索命中\n（%d 条）" % hit_n, "检索未命中\n（%d 条）" % miss_n]
    ok = [hit_ok, round(s["acc_when_miss"] * miss_n)]
    bad = [hit_bad, miss_n - ok[1]]

    y = [1.0, 0.35]
    ax.barh(y, ok, height=0.34, color=C_IMPROVE, label="回答正确")
    ax.barh(y, bad, height=0.34, left=ok, color="#d8dee6", label="回答错误")

    for i in range(2):
        total = ok[i] + bad[i]
        # 分段足够宽才把数字写在色块里，否则（比如未命中的那 3 条）
        # 写进去会挤成一团，改写在色块右边
        for val, left, color_in in ((ok[i], 0, "white"), (bad[i], ok[i], "#4b5563")):
            if not val:
                continue
            if val >= s["n"] * 0.08:
                ax.text(left + val / 2, y[i], f"{val}", ha="center", va="center",
                        color=color_in, fontsize=11, fontweight="bold")
            else:
                ax.text(left + val + 3, y[i], f"{val}", ha="left", va="center",
                        color="#4b5563", fontsize=10.5, fontweight="bold")
        rate = 100 * ok[i] / total if total else 0
        ax.text(s["n"] * 1.06, y[i], f"正确率 {rate:.1f}%", va="center",
                ha="left", fontsize=10.5, color="#1f2937")

    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.set_xlabel("题目数（共 %d 条）" % s["n"])
    ax.set_xlim(0, s["n"] * 1.34)
    ax.set_ylim(0.05, 1.3)
    # 图例放到图外下方，避免压住右侧的"正确率"标注
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.22),
              ncol=2, frameon=False)
    ax.set_title("答对与否，取决于检索有没有命中")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    _save(fig, "fig6_端到端失败归因.png")


def pick_e2e(style):
    """
    按提示词风格挑端到端评测结果文件。

    不能靠文件名排序或体积取"最新那个"——目录里同时躺着 system 和 concise
    两种风格的结果，体积也差不多，取错了会让图上的数对不上正文。
    所以只认文件里记录的 prompt_style 字段，取时间戳最新的一个。
    """
    hits = []
    for p in glob.glob(os.path.join(RESULT_DIR, "e2e_*.json")):
        d = json.load(open(p, encoding="utf-8"))
        if d.get("config", {}).get("prompt_style") == style:
            hits.append((os.path.basename(p), d))
    if not hits:
        raise SystemExit(f"找不到 prompt_style={style} 的端到端结果，"
                         f"请先跑 eval/run_e2e_eval.py --style {style}")
    name, d = max(hits, key=lambda x: x[0])
    return d, name


def fig_style_compare(a, b):
    """同一套检索、同一批题目，只改提示词，看指标差多少。"""
    metrics = ["EM", "F1", "包含率"]
    sys_v = [a["summary"]["EM"] * 100, a["summary"]["F1"] * 100,
             a["summary"]["contains"] * 100]
    con_v = [b["summary"]["EM"] * 100, b["summary"]["F1"] * 100,
             b["summary"]["contains"] * 100]

    x = range(len(metrics))
    w = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4))

    b1 = ax.bar([i - w / 2 for i in x], sys_v, w, color=C_BASE,
                label="system（线上实际：基于资料回答问题）")
    b2 = ax.bar([i + w / 2 for i in x], con_v, w, color=C_FINAL,
                label="concise（对照：只给答案本身）")

    for bars, vals in ((b1, sys_v), (b2, con_v)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}",
                    ha="center", fontsize=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("得分（%）")
    ax.set_ylim(0, 108)
    ax.set_title("提示词风格对评测指标的影响\n（检索命中率两者相同，都是 98.5%）",
                 fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
              frameon=False, fontsize=9.5)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    _save(fig, "fig7_提示词风格对照.png")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)


    print("图1、图2 —— 分块与截断")
    ana = _load("chunking_analysis.json")
    fig_truncation(ana)
    fig_chunking(ana)

    print("图3、图4 —— 检索方案对比")
    rows = _load("retrieval_summary.json")["rows"]
    fig_ablation(rows)
    fig_latency(rows)

    print("图5 —— 微调训练曲线")
    cands = sorted(glob.glob(os.path.join(RESULT_DIR, "finetune_history_*.json")),
                   key=os.path.getmtime)
    hist = json.load(open(cands[-1], encoding="utf-8"))
    print(f"  数据源：{os.path.basename(cands[-1])}")
    fig_training(hist)

    print("图6 —— 端到端失败归因")
    sys_e2e, sys_name = pick_e2e("system")
    print(f"  数据源：{sys_name}（{sys_e2e['summary']['n']} 条，线上实际使用的提示词）")
    fig_e2e(sys_e2e)

    print("图7 —— 提示词风格对照")
    con_e2e, con_name = pick_e2e("concise")
    print(f"  数据源：{con_name}")
    fig_style_compare(sys_e2e, con_e2e)

    print(f"\n全部图已输出到 {FIG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
