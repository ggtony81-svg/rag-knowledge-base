# -*- coding: utf-8 -*-
"""
文档分块 —— 评测侧入口

背景：
    检索对比实验发现，向量检索（P@1=0.8590）明显不如 BM25 关键词检索
    （P@1=0.9164）。进一步统计发现评测语料平均 582 个 token，而
    bge-small-zh-v1.5 的最大输入长度只有 512 个 token，63.5% 的文档在
    编码时会被截断；BM25 却是基于完整文档做词频统计的。因此推测
    "长文档被截断"是向量检索落后的主要原因之一。

    如果这个推测成立，那么把长文档先切成短块、再对块做向量检索，
    应当能明显改善效果——每个块都能完整放进模型，不再丢信息。

实现放在仓库根目录的 text_chunk.py，由线上服务 pdf_qa_optimized.py
和本目录共同 import。这样保证"评测跑的分块"和"线上跑的分块"是同一份代码，
评测结论可以直接迁移到线上。

    1. smart_chunk     —— 按"第X章 / 第X条"结构标记切分
    2. smart_chunk_v2  —— 在其基础上增加"按长度切分"，对长文档也生效

本文件只保留"评测入口 + 自检"，实现不再重复一份。
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 实现放在仓库根目录，需要把上一级目录加进来才能 import。
# 必须放在下面的 import 之前。
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from text_chunk import (            # noqa: E402
    MODEL_MAX_TOKENS,
    get_tokenizer,
    n_tokens,
    smart_chunk,
    smart_chunk_v2,
    split_sentences,
    _SECTION_PREFIXES,
    _tail,
)

__all__ = [
    "MODEL_MAX_TOKENS", "get_tokenizer", "n_tokens",
    "smart_chunk", "smart_chunk_v2", "split_sentences",
]


# ---------------------------------------------------------------------------
# 自检：对比两种分块方式在本项目语料上的实际表现
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import statistics

    import dataset

    tok = get_tokenizer()
    corpus, _, _ = dataset.build_retrieval_benchmark(dedup=True)
    texts = corpus["text"].tolist()

    print(f"语料文档数：{len(texts)}")
    print()

    for name, fn in [("smart_chunk（原系统）", smart_chunk),
                     ("smart_chunk_v2（改进）", smart_chunk_v2)]:
        counts, toks, over = [], [], 0
        for t in texts:
            cs = fn(t)
            counts.append(len(cs))
            for c in cs:
                n = n_tokens(c, tok)
                toks.append(n)
                if n > MODEL_MAX_TOKENS:
                    over += 1
        total = sum(counts)
        print(f"{name}")
        print(f"  总块数            : {total}")
        print(f"  平均每篇块数      : {statistics.mean(counts):.2f}")
        print(f"  未切分的文档数    : {sum(1 for c in counts if c == 1)} / {len(texts)}")
        print(f"  块长度 平均/最大  : {statistics.mean(toks):.0f} / {max(toks)} token")
        print(f"  超过 {MODEL_MAX_TOKENS} 会被截断的块 : {over} ({over / total * 100:.1f}%)")
        print()
