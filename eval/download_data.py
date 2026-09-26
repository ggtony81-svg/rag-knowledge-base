# -*- coding: utf-8 -*-
"""
数据集下载脚本

从 HuggingFace 镜像下载本项目用到的三个公开数据集，保存到 eval/data/。

数据来源：
  1. hfl/cmrc2018
     哈工大讯飞联合实验室发布的中文机器阅读理解数据集（CMRC 2018）。
     用于：微调训练集、端到端问答评测集。
     主页：https://huggingface.co/datasets/hfl/cmrc2018

  2. jinaai/longcontext-cmrc2018-zh
     把 CMRC2018 转成 BEIR 检索格式后的长文本版本，含语料库(corpus)与查询(queries)。
     用于：检索评测的知识库语料。
     主页：https://huggingface.co/datasets/jinaai/longcontext-cmrc2018-zh

  3. jinaai/longcontext-cmrc2018-zh-qrels
     上述数据集的标准答案，标注了每条查询对应哪篇文档是相关的。
     用于：计算 Recall@k / MRR / nDCG 等检索指标。
     主页：https://huggingface.co/datasets/jinaai/longcontext-cmrc2018-zh-qrels

国内直连 huggingface.co 通常不通，因此走 hf-mirror.com 镜像。
"""
import os
import subprocess
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MIRROR = "https://hf-mirror.com"

# (仓库名, 仓库内文件路径, 保存到本地的文件名)
FILES = [
    ("hfl/cmrc2018", "data/train-00000-of-00001.parquet", "cmrc_train.parquet"),
    ("hfl/cmrc2018", "data/validation-00000-of-00001.parquet", "cmrc_val.parquet"),
    ("jinaai/longcontext-cmrc2018-zh", "data/corpus-00000-of-00001.parquet", "lc_corpus.parquet"),
    ("jinaai/longcontext-cmrc2018-zh", "data/queries-00000-of-00001.parquet", "lc_queries.parquet"),
    ("jinaai/longcontext-cmrc2018-zh-qrels", "data/dev-00000-of-00001.parquet", "lc_qrels.parquet"),
]


def download(repo, remote_path, local_name, force=False):
    out_path = os.path.join(DATA_DIR, local_name)
    if os.path.exists(out_path) and not force:
        print(f"  已存在，跳过：{local_name} ({os.path.getsize(out_path)} 字节)")
        return True

    url = f"{MIRROR}/datasets/{repo}/resolve/main/{remote_path}"
    print(f"  下载 {repo} -> {local_name}")
    # 用 curl 而不是 huggingface_hub：镜像对 datasets 类型的直链更稳定
    result = subprocess.run(
        ["curl", "-sL", "--fail", "--max-time", "600", "-o", out_path, url],
        capture_output=True,
    )
    if result.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
        print(f"  下载失败：{url}")
        print(f"  curl 返回码 {result.returncode}，可以手动下载后放到 {DATA_DIR}")
        return False
    print(f"  完成：{os.path.getsize(out_path)} 字节")
    return True


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    force = "--force" in sys.argv
    print(f"数据保存目录：{DATA_DIR}")
    ok = all([download(r, p, n, force) for r, p, n in FILES])
    if ok:
        print("\n全部数据就绪。运行 python eval/dataset.py 可查看数据统计。")
    else:
        print("\n部分数据下载失败，请检查网络。")
        sys.exit(1)


if __name__ == "__main__":
    main()
