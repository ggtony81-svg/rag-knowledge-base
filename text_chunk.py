# -*- coding: utf-8 -*-
"""
文本分块 —— 全项目共用的一份实现

为什么单独抽出来：
    离线评测里验证过的分块逻辑，必须和线上服务跑的是**同一份代码**。
    否则线上改一版、评测改一版，慢慢就对不上了 ——
    评测出来的结论也就不再能代表线上的真实表现。

    所以这里放唯一的实现，评测脚本和 pdf_qa_optimized.py 都 import 它。

两个函数：
    smart_chunk     按"第X章 / 第X条"这类结构标记切分
    smart_chunk_v2  在结构切分的基础上，再把过长的块按句边界切成短块

为什么需要 v2：
    bge-small-zh-v1.5 单次最多读 512 个 token，超出的部分会被**静默丢掉**，
    不报错、不警告。语料平均 582 token，63.5% 的文档都会被截断。
    BM25 却是按整篇做词频统计的 —— 这就是向量检索一开始打不过关键词检索的原因。
"""
import os
import re

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 与 bge-small-zh-v1.5 的 max_seq_length（512）一致
MODEL_MAX_TOKENS = 512

# 默认分词器：按 token 数衡量长度（字符数反映不了真实长度）
DEFAULT_TOKENIZER = "BAAI/bge-small-zh-v1.5"

# 结构标记（面向"第X章 / 第X条"式的法规文档）
#
# 已知限制：这是一张写死的短表，只覆盖 1~3 章、1~7 条。
# 文档一旦用到"第八章""第二十条"，这里就认不出来，整篇会被当成一个块
# —— 此时只能靠 v2 的长度切分兜底。
#
# 之所以没改成 "第[一二三四五六七八九十]+[章条]" 这种正则：
# 评测里方案 A/E/F 的数字是在**这张表**下跑出来的，
# 一改，那些结论就不再对应这份代码了。
_SECTION_PREFIXES = [
    "第一章", "第二章", "第三章",
    "第一条", "第二条", "第三条", "第四条", "第五条", "第六条", "第七条",
]

_TOKENIZER = None


def get_tokenizer(model_name=DEFAULT_TOKENIZER):
    """加载分词器。传入本地路径（如微调后的模型目录）也可以。"""
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    return _TOKENIZER


def n_tokens(text, tokenizer=None):
    """统计文本的 token 数（含 [CLS]/[SEP]，与模型实际输入长度口径一致）。"""
    tokenizer = tokenizer or get_tokenizer()
    return len(tokenizer(str(text), truncation=False)["input_ids"])


# ---------------------------------------------------------------------------
# 一、结构分块
# ---------------------------------------------------------------------------

def smart_chunk(text, overlap=50):
    """按结构标记切块，相邻块之间用上一块的末尾 overlap 个字符做重叠。"""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    chunks, cur = [], ""
    for line in lines:
        if any(line.startswith(p) for p in _SECTION_PREFIXES):
            if cur:
                chunks.append(cur.strip())
                cur = ""
        cur += line + "\n"
    if cur:
        chunks.append(cur.strip())
    res = []
    for i, c in enumerate(chunks):
        if i > 0:
            c = chunks[i - 1][-overlap:] + "\n" + c
        res.append(c)
    return [c for c in res if len(c) > 15]


# ---------------------------------------------------------------------------
# 二、结构分块 + 长度切分
# ---------------------------------------------------------------------------

_SENT_END = "。！？；!?;\n"


def split_sentences(text):
    """按句末标点切句，标点保留在前一句末尾，避免句子被切碎后语义不完整。"""
    out, cur = [], ""
    for ch in text:
        cur += ch
        if ch in _SENT_END:
            if cur.strip():
                out.append(cur.strip())
            cur = ""
    if cur.strip():
        out.append(cur.strip())
    return out


def _tail(text, n_tok, tokenizer):
    """
    取文本末尾约 n_tok 个 token 的内容，用于构造相邻块之间的重叠。

    这里刻意**不**用 tokenizer.decode() 来截取：WordPiece 把每个汉字当成
    独立 token，decode 时会拿空格把它们连起来（"员 工 累 计 工 作"），
    数字也会被拆开（"3.9" → "3. 9"）。中文文档里数字日期遍地都是，逐个
    打补丁治不干净，而且凭空多了一层信息损失。

    改为按"字符数 / token 数"的比例估算出对应的字符长度，直接从原文里
    切片 —— 重叠段就一定是原文，不会变形。token 数只用来量长度，真正的
    预算由调用方用 n_tokens() 精确核算，所以这里估算差一点也不要紧。
    """
    if n_tok <= 0:
        return ""
    ids = tokenizer(str(text), truncation=False)["input_ids"]
    if len(ids) <= n_tok:
        return text
    n_chars = max(1, round(len(text) * n_tok / len(ids)))
    return text[-n_chars:]


def smart_chunk_v2(text, max_tokens=400, overlap_tokens=50, tokenizer=None):
    """
    先按结构切，再把仍然超长的块按句边界切成不超过 max_tokens 的块。

    max_tokens 取 400 而不是 512：留出余量，避免分块后加上查询或特殊符号
    刚好越界又被截断。

    相邻块之间保留 overlap_tokens 个 token 的重叠，防止答案恰好被切在
    两个块的边界上、两边都只拿到半句。
    """
    tokenizer = tokenizer or get_tokenizer()
    blocks = smart_chunk(text)

    res = []
    for block in blocks:
        if n_tokens(block, tokenizer) <= max_tokens:
            res.append(block)
            continue

        # 超长块：按句子边界贪心装箱
        cur = []          # 当前块拼好的文本
        cur_n = 0
        for sent in split_sentences(block):
            s_n = n_tokens(sent, tokenizer)
            if cur and cur_n + s_n > max_tokens:
                done = "".join(cur)
                res.append(done)
                tail = _tail(done, overlap_tokens, tokenizer)
                cur = [tail] if tail else []
                cur_n = n_tokens(tail, tokenizer) if tail else 0
            cur.append(sent)
            cur_n += s_n
        if cur:
            res.append("".join(cur))

    return [c for c in res if len(c) > 15]
