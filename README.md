# RAG 知识库问答系统

上传 PDF 自动建库；提问时先检索出最相关的段落，再交给大模型**仅依据这些段落**作答，答案流式返回。

这个仓库的重点不是"跑通了一个 RAG"，而是：**用一套离线评测体系，把检索方案的选择从"感觉"变成"数据"**。

---

## 一段话结论

通用向量模型 `bge-small-zh-v1.5` 在 CMRC2018 上的 P@1 只有 **85.9%**，**打不过关键词检索 BM25 的 91.6%**。

排查定位到两个原因：

1. **领域不匹配** —— 通用语料预训练的模型没见过本领域"问题 → 段落"的对应关系；
2. **静默截断** —— 语料平均 582 token，超出模型 512 的上限，**63.5% 的文档被截掉后半段，且不报任何错**。

分别修复后（结构分块 + 领域微调），P@1 达到 **92.4%**，与"向量 + CrossEncoder 重排序"方案**完全持平**，但单次检索从 **6345ms 降到 17ms（365 倍）**。

线上最终采用后者。

---

## 评测驱动的检索选型

**评测集**：CMRC2018（哈工大讯飞中文机器阅读理解）—— 1149 条查询 / 934 篇语料，
每条查询有唯一的标准相关文档。指标按 BEIR 口径计算。

| 方案 | P@1 | Recall@3 | Recall@10 | MRR@10 | nDCG@10 | 单次耗时 |
|---|---|---|---|---|---|---|
| A · BM25 关键词 | 0.9164 | 0.9721 | 0.9878 | 0.9455 | 0.9561 | 7.1 ms |
| B · 向量检索 | 0.8590 | 0.9017 | 0.9460 | 0.8863 | 0.9007 | 18.1 ms |
| C · 向量 + 重排序 | **0.9243** | 0.9399 | 0.9460 | 0.9327 | 0.9360 | 6345.7 ms |
| D · 向量检索（微调） | 0.8895 | 0.9356 | 0.9669 | 0.9158 | 0.9283 | 21.9 ms |
| E · 分块 + 向量检索 | 0.9051 | 0.9434 | 0.9634 | 0.9262 | 0.9353 | 16.7 ms |
| **F · 分块 + 微调** | **0.9243** | **0.9600** | **0.9861** | **0.9445** | **0.9545** | **17.4 ms** |

三个值得说的点：

- **B < A 是反直觉的**。语义检索通常被认为强于关键词检索，但在这里输给 BM25 近 6 个百分点。
  如果只看 B 就下"向量检索更好/更差"的结论，方向就错了——问题出在数据没有适配模型，而不是模型不好。
- **E、D 分别验证了两个原因**。分块单独带来 +4.6pt（B→E），微调单独带来 +3.0pt（B→D），
  两者叠加后是 +6.5pt（B→F），**说明这两个原因基本独立，都要修**。
- **F 和 C 打平**。这是选型的关键：重排序能拿到的效果，通过"先把数据准备好 + 让模型见过本领域数据"也能拿到，
  代价却小两个数量级。**所以线上不需要重排序。**

> 完整数据：`eval/results/retrieval_summary.md`
> 逐次运行的原始结果：`eval/results/retrieval_*.json`

### 端到端评测（200 条）

检索对了，大模型就一定能答对吗？单独测了一次：

| | |
|---|---|
| 检索命中标准段落 | 197 / 200（**98.5%**） |
| &nbsp;&nbsp;└ 其中答对 | **88.8%** |
| 检索未命中 | 3 / 200（1.5%） |
| &nbsp;&nbsp;└ 其中答对 | **0.0%** |

**检索没命中时答对率是 0** —— 说明这个系统的瓶颈在检索，不在生成。
这决定了优化方向：继续调 Prompt 收益有限，应该继续投入检索。

### 避免指标虚高

微调前检查了训练集与评测集的重叠：CMRC2018 训练集里有 **204 篇文档同时出现在评测语料中**，
剔除后才开始训练。不剔的话模型相当于做过这些题，指标会好看但不可信。

---

## 两个工程决策

### 1. 为什么线上没有用重排序

CrossEncoder 重排序确实有用（C 比 B 高 6.5pt），但评测显示方案 F 已经达到同样的 P@1，
而检索耗时是 **17ms vs 6345ms**。为了一个已经拿到的指标付出 365 倍延迟，不划算。

如果将来语料换成领域差异大、F 方案掉下来的场景，重排序仍是值得重新考虑的选项 ——
代码在 `eval/retrievers.py` 里保留着。

### 2. 知识库落盘要带"指纹"

向量库序列化到磁盘时，会一并记录**生成它的模型路径、分块参数、向量维度**。
启动加载时逐项比对，不一致就拒绝加载并提示重新上传。

原因：换了 embedding 模型却沿用旧向量，相似度**照样算得出来**（维度都是 512），
只是算出来的是错的 —— 不报错、不崩溃，静默给出错误排序。
这类"错误伪装成成功"的问题比崩溃难查得多。

---

## 架构

```
上传阶段
  PDF ──PyMuPDF──▶ 纯文本 ──smart_chunk_v2──▶ 文本块 ──微调模型──▶ 向量 ──▶ 落盘
                            (400 token 上限               (bge-small-zh
                             / 50 token 重叠)              + 领域微调)

问答阶段
  问题 ──▶ Redis 缓存 ──命中──▶ 直接返回（不调模型）
              │未命中
              ▼
         向量化 ──▶ 与全部块做矩阵乘 ──▶ Top-3 ──▶ 拼进 Prompt ──▶ DeepSeek 流式生成
                                                                      │
                                              写回 Redis 缓存 ◀───────┘
                                              写入 MySQL 会话历史
```

**MySQL 和 Redis 不可用时全部走降级分支**（缓存直查、锁放行），主流程不受影响。

---

## 目录结构

```
rag-knowledge-base/
├── pdf_qa_optimized.py     # 线上主程序（FastAPI，端口 18005）
├── text_chunk.py           # 分块实现 —— 线上与 eval/ 共用同一份，杜绝两边漂移
├── agent_studio_api.py     # 另一个项目：多智能体协作系统后端
├── studio.html             # 多智能体系统的前端页面
├── generate_test_pdf.py    # 生成测试用的中文 PDF（9 章 39 条，用来验证分块）
│
├── eval/                   # 离线评测体系
│   ├── download_data.py    # 下载 CMRC2018 数据集
│   ├── chunking.py         # 分块的兼容层，实际实现已抽到根目录 text_chunk.py
│   ├── retrievers.py       # 6 种检索方案的实现（BM25 / 向量 / 重排序 / 微调 / 分块）
│   ├── metrics.py          # P@k / Recall@k / MRR / nDCG
│   ├── finetune_embedding.py   # 领域微调（批内负样本对比学习）
│   ├── run_retrieval_eval.py   # 检索方案对比
│   ├── run_e2e_eval.py         # 端到端问答评测 + 失败归因
│   └── results/            # 评测结果（表格、日志、逐次运行的原始 JSON）
│
├── web/                    # 问答界面（原生 HTML + CSS + JS，无框架）
│
├── config.example.py       # 配置模板 → 复制为 config.py 并填 API Key
├── requirements.txt
└── learning/               # 学习过程中的实验与历史版本，非本项目主线
    ├── rag_api.py … pdf_qa_redis.py   # v1 → v6 的演进版本
    ├── *_demo.py                      # 各环节原理 demo（Agent / 流式 / Redis …）
    └── *.md                           # 学习笔记与讲解文档
```

---

## 快速开始

### 1. 安装

```bash
git clone https://github.com/ggtony81-svg/rag-knowledge-base.git
cd rag-knowledge-base
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 配置

```bash
cp config.example.py config.py
# 编辑 config.py，填入你的 DEEPSEEK_API_KEY
```

### 3. 准备检索模型

线上默认加载**微调后**的模型（`eval/cache/finetuned_model`）。仓库不含模型权重（93MB），两种方式二选一：

```bash
# 方式一：先跑起来看效果 —— 用通用模型，P@1 85.9%
MODEL_PATH=BAAI/bge-small-zh-v1.5 python pdf_qa_optimized.py

# 方式二：完整复现评测结论 —— 微调，CPU 上约 3.6 小时
python eval/download_data.py        # 下载 CMRC2018（约 5.7MB）
python eval/finetune_embedding.py   # 微调并保存到 eval/cache/finetuned_model
```

> 为什么不做"找不到微调模型就自动退回通用模型"的兜底？
> 那样指标会从 92.4% 悄悄掉到 85.9% 而没有任何提示。**宁可启动报错，也不要静默降级。**

### 4. 启动

```bash
python pdf_qa_optimized.py
# 打开 http://127.0.0.1:18005
```

MySQL 和 Redis 是可选的，没有也能跑（会自动降级）。想启用缓存，本机需要：

```bash
# MySQL：建库 pdf_qa 即可，表结构首次启动自动创建
# Redis：默认 localhost:6379
```

没有现成的 PDF 可以试？先造一份测试文档：

```bash
python generate_test_pdf.py    # 生成 test_handbook.pdf（星海科技员工手册）
```

---

## 复现评测

```bash
# 可选方案：bm25 / dense / dense_rerank / bm25_rerank / dense_chunk / dense_ft / dense_chunk_ft
python eval/run_retrieval_eval.py --schemes bm25 dense dense_rerank dense_ft dense_chunk dense_chunk_ft
python eval/summarize_results.py        # 生成 eval/results/retrieval_summary.md
python eval/run_e2e_eval.py             # 端到端评测（200 条，约 10 分钟）
```

---

## 技术栈

| 环节 | 技术 |
|---|---|
| 后端 | Python · FastAPI · SSE 流式输出 |
| 大模型 | DeepSeek API（OpenAI 兼容格式） |
| 检索 | bge-small-zh-v1.5（512 维，领域微调）· BM25 · NumPy 矩阵运算 |
| 微调 | sentence-transformers · MultipleNegativesRankingLoss |
| 文档解析 | PyMuPDF |
| 存储 | MySQL（会话与消息，外键级联）· Redis（Cache-Aside，含降级） |
| 前端 | 原生 HTML + CSS + JS，无框架 |

---

## 版本演进

项目的完整演进过程（v1 关键词 → v6 工程化 → v7 评测驱动优化）
和每个环节的原理 demo 都在 [`learning/`](learning/) 目录下。

---

## License

MIT
