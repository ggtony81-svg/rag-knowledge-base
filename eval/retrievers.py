# -*- coding: utf-8 -*-
"""
检索器模块

本项目要做的是"文档问答"，整条链路的第一步是"根据问题找到相关文档"。
这一步的效果直接决定最终回答的质量，因此需要单独评测。

本模块实现四种检索方案，用于做对比实验（消融实验）：

    方案A  BM25Retriever         关键词检索（传统信息检索基线）
    方案B  DenseRetriever        向量语义检索（bge-small-zh-v1.5）
    方案C  RerankRetriever       向量召回 + 交叉编码器重排序
    方案D  DenseRetriever(微调)  用微调后的向量模型做检索
    方案E  ChunkedDenseRetriever 先分块、再对块做向量检索

各方案对外接口完全一致：index() 建索引，search() 返回排序结果，
这样评测脚本可以用同一套代码跑完所有方案，保证对比公平。
"""
import os

import numpy as np

# 国内直连 huggingface.co 不通，走镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 本机为 6 核 12 线程，PyTorch 默认只用一半线程。这里显式用满：
# 实测重排序吞吐从 2.2 对/秒提升到 2.4 对/秒（该模型在 CPU 上本就受算力限制，
# 线程数不是主要瓶颈，但设满没有坏处）。必须在 import torch 之前设置才生效。
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 8))

try:
    import torch
    torch.set_num_threads(os.cpu_count() or 8)
except ImportError:
    pass

EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"

# bge 系列模型官方建议：检索任务中给"查询"加上指令前缀，文档侧不加。
# 注意：原项目 pdf_qa_optimized.py 没有加这个前缀，所以这里默认关闭，
# 保证"方案B"复现的是原系统的真实行为；打开后的效果单独作为一组实验记录。
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class BaseRetriever:
    """检索器统一接口。子类只需实现 index() 和 search()。"""

    name = "base"

    def index(self, doc_ids, doc_texts):
        raise NotImplementedError

    def search(self, query, topk=10):
        """返回 [(doc_id, score), ...]，按 score 从高到低排序。"""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 方案A：BM25 关键词检索
# ---------------------------------------------------------------------------

class BM25Retriever(BaseRetriever):
    """
    BM25 关键词检索。

    BM25 是信息检索里的经典排序函数，本质是"词频 + 逆文档频率"的加权，
    并对"文档过长"和"词频过高"做了饱和处理：

        score(D, Q) = Σ  IDF(qi) · f(qi,D)·(k1+1) / ( f(qi,D) + k1·(1-b+b·|D|/avgdl) )
        IDF(qi)     = ln( 1 + (N - n(qi) + 0.5) / (n(qi) + 0.5) )

    其中 f(qi,D) 是词 qi 在文档 D 中的出现次数，|D| 是文档长度，
    avgdl 是平均文档长度，N 是文档总数，n(qi) 是包含 qi 的文档数。

    参数取经验默认值 k1=1.5、b=0.75。

    中文需要先分词，这里用 jieba。

    实现说明：打分时只遍历"包含查询词"的文档，而不是遍历整个语料库。
    为此在 index() 阶段建立倒排索引 postings[term] = [(doc_idx, 词频), ...]。
    1149 篇文档的规模下，这能把单条查询的打分从"上千次扫描"降到"几十次命中"。
    """

    name = "BM25"

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids = []
        self.doc_len = None
        self.avgdl = 0.0
        self.postings = {}    # 词 -> [(文档下标, 该词在文档中的词频), ...]
        self.N = 0

    @staticmethod
    def tokenize(text):
        import jieba
        return [w for w in jieba.lcut(str(text)) if w.strip()]

    def index(self, doc_ids, doc_texts):
        self.doc_ids = list(doc_ids)
        self.N = len(self.doc_ids)
        self.postings = {}
        lengths = []

        for i, text in enumerate(doc_texts):
            tokens = self.tokenize(text)
            lengths.append(len(tokens))
            tf = {}
            for w in tokens:
                tf[w] = tf.get(w, 0) + 1
            for w, f in tf.items():
                self.postings.setdefault(w, []).append((i, f))

        self.doc_len = np.array(lengths, dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.N else 0.0

    def _idf(self, term):
        """IDF(qi) = ln( 1 + (N - n(qi) + 0.5) / (n(qi) + 0.5) )"""
        n = len(self.postings.get(term, ()))
        return float(np.log(1.0 + (self.N - n + 0.5) / (n + 0.5)))

    def search(self, query, topk=10):
        q_terms = self.tokenize(query)
        if not q_terms:
            return []

        scores = np.zeros(self.N, dtype=np.float32)
        for term in q_terms:
            posting = self.postings.get(term)
            if not posting:      # 该词不在任何文档中，IDF 再大也贡献不了分数
                continue
            idf = self._idf(term)
            for i, f in posting:
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1.0))
                scores[i] += idf * f * (self.k1 + 1) / denom

        order = np.argsort(-scores)[:topk]
        return [(self.doc_ids[i], float(scores[i])) for i in order if scores[i] > 0]


# ---------------------------------------------------------------------------
# 方案B / D：向量语义检索
# ---------------------------------------------------------------------------

class DenseRetriever(BaseRetriever):
    """
    向量语义检索。

    做法：把每篇文档编码成一个固定长度的向量（bge-small-zh-v1.5 输出 512 维），
    查询同样编码成向量，用余弦相似度衡量"意思接近程度"，取最相似的 topk 篇。

    余弦相似度：
        cos(q, d) = (q · d) / (|q| · |d|)

    因为 bge 输出的向量已经做过 L2 归一化，实际计算时内积就等于余弦相似度。
    这里仍然显式归一化一次，保证换模型时结果依然正确。

    微调后的模型只是换了 model_name_or_path，其余逻辑完全一致，
    这样"微调前 vs 微调后"的对比就只差模型本身这一个变量。
    """

    name = "Dense"

    def __init__(self, model_name=EMBED_MODEL_NAME, use_instruction=False,
                 device=None, cache_dir=None):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.use_instruction = use_instruction
        self.model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir)
        self.doc_ids = []
        self.doc_vecs = None

    def _encode(self, texts, is_query=False):
        if is_query and self.use_instruction:
            texts = [QUERY_INSTRUCTION + t for t in texts]
        vecs = self.model.encode(texts, batch_size=32, show_progress_bar=False,
                                 convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

    def encode_docs(self, doc_texts):
        return self._encode(list(doc_texts), is_query=False)

    def index(self, doc_ids, doc_texts, doc_vecs=None):
        self.doc_ids = list(doc_ids)
        self.doc_vecs = doc_vecs if doc_vecs is not None else self.encode_docs(doc_texts)

    def search(self, query, topk=10):
        qv = self._encode([query], is_query=True)[0]
        scores = self.doc_vecs @ qv          # 已归一化，内积即余弦相似度
        order = np.argsort(-scores)[:topk]
        return [(self.doc_ids[i], float(scores[i])) for i in order]


# ---------------------------------------------------------------------------
# 方案E：先分块再向量检索
# ---------------------------------------------------------------------------

class ChunkedDenseRetriever(BaseRetriever):
    """
    先分块、再对块做向量检索。

    动机：评测语料平均 582 个 token，超过 bge-small-zh-v1.5 的 512 上限，
    实测 63.5% 的文档在编码时会被截断，而 BM25 是基于完整文档统计词频的。
    把长文档切成不超过 max_tokens 的文本块后，每个块都能完整编码，不再丢信息。

    检索流程：
        1. 建索引时先把每篇文档切成若干块，对"块"而不是"文档"编码
        2. 检索时用查询向量和所有块算相似度
        3. 聚合回文档级别：一篇文档的得分 = 它所有块中的最高分
           （即"只要有一个块跟问题相关，这篇文档就算相关"）

    用最高分而不是平均分聚合，是因为一篇文档通常只有一小段跟问题相关，
    取平均会把那段的关键信息稀释掉。
    """

    name = "ChunkedDense"

    def __init__(self, model_name=EMBED_MODEL_NAME, max_tokens=400,
                 overlap_tokens=50, use_instruction=False):
        self.dense = DenseRetriever(model_name=model_name, use_instruction=use_instruction)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.doc_ids = []
        self.chunks = []
        self.chunk_doc_idx = None    # 每个块属于第几篇文档（下标）
        self.chunk_vecs = None

    def index(self, doc_ids, doc_texts, doc_vecs=None):
        from chunking import smart_chunk_v2   # 延迟导入，避免与 chunking 模块循环引用

        self.doc_ids = list(doc_ids)
        self.chunks = []
        self.chunk_doc_idx = []
        for di, text in enumerate(doc_texts):
            for c in smart_chunk_v2(text, max_tokens=self.max_tokens,
                                    overlap_tokens=self.overlap_tokens):
                self.chunks.append(c)
                self.chunk_doc_idx.append(di)

        self.chunk_doc_idx = np.asarray(self.chunk_doc_idx, dtype=np.int64)
        self.chunk_vecs = self.dense.encode_docs(self.chunks)

    def search(self, query, topk=10):
        qv = self.dense._encode([query], is_query=True)[0]
        chunk_scores = self.chunk_vecs @ qv

        # 聚合到文档级别：取该文档所有块中的最高分
        doc_scores = np.full(len(self.doc_ids), -np.inf, dtype=np.float32)
        np.maximum.at(doc_scores, self.chunk_doc_idx, chunk_scores)

        order = np.argsort(-doc_scores)[:topk]
        return [(self.doc_ids[i], float(doc_scores[i])) for i in order]


# ---------------------------------------------------------------------------
# 方案C：向量召回 + 重排序
# ---------------------------------------------------------------------------

class RerankRetriever(BaseRetriever):
    """
    召回 + 交叉编码器重排序。

    第一阶段是"召回"：用一种便宜的方法（向量检索或 BM25）快速从整个语料库
    里挑出一小批候选文档。向量检索是"双塔"结构，查询和文档各自独立编码成向量，
    可以预先算好、速度极快，但两边在编码时互相看不见，细粒度差异容易丢失。

    第二阶段是"精排"：交叉编码器（CrossEncoder）把"查询+文档"拼成一个句子对
    整体送进模型，让两者充分交互后输出相关性分数，判断更准；代价是每篇文档都要
    单独跑一次前向传播，无法预计算，所以只能用来处理召回阶段的小候选集。

    recall_k 是"召回率"与"耗时"之间的折中：候选越多越不容易漏掉正确文档，
    但重排序耗时线性增长。

    base_kind 用于指定召回阶段用哪种检索器，默认向量检索；
    由于实验中 BM25 表现更好，也支持用 BM25 做召回，以便对比。
    """

    name = "Rerank"

    def __init__(self, base_kind="dense", model_name=EMBED_MODEL_NAME,
                 rerank_model_name=RERANK_MODEL_NAME, recall_k=50,
                 use_instruction=False, max_length=512):
        self.recall_k = recall_k
        self.base_kind = base_kind
        if base_kind == "bm25":
            self.base = BM25Retriever()
        else:
            self.base = DenseRetriever(model_name=model_name, use_instruction=use_instruction)

        from sentence_transformers import CrossEncoder
        self.reranker = CrossEncoder(rerank_model_name)
        # 本项目语料平均约 620 字，加查询后约 400 个 token，512 可完整覆盖不截断。
        # 实测：max_length=512 时约 2.4 对/秒，降到 256 可提速到约 4.9 对/秒，
        # 但会把长文档截断、影响精排质量，因此默认保持 512。
        self.reranker.max_length = max_length
        self._id2text = {}

    def index(self, doc_ids, doc_texts, doc_vecs=None):
        self.doc_ids = list(doc_ids)
        self._id2text = dict(zip(doc_ids, doc_texts))
        if isinstance(self.base, BM25Retriever):
            self.base.index(doc_ids, doc_texts)
        else:
            self.base.index(doc_ids, doc_texts, doc_vecs=doc_vecs)

    def search(self, query, topk=10):
        cand = self.base.search(query, topk=self.recall_k)
        if not cand:
            return []
        pairs = [(query, self._id2text[d]) for d, _ in cand]
        re_scores = self.reranker.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            ((cand[i][0], float(re_scores[i])) for i in range(len(cand))),
            key=lambda x: -x[1],
        )
        return ranked[:topk]


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def build_retriever(kind, **kwargs):
    """按名称构造检索器，供评测脚本统一调用。"""
    kind = kind.lower()
    if kind == "bm25":
        return BM25Retriever()
    if kind == "dense":
        return DenseRetriever(**kwargs)
    if kind == "dense_rerank":
        return DenseRerankRetriever(**kwargs)
    raise ValueError(f"未知的检索器类型：{kind}")
