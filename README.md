# 📚 企业知识库 RAG 问答系统 — 从零到一的 AI 学习之旅

> 这不是一次性做出来的产品，**而是一步一步"学"出来的作品**。
>
> 从完全不会 AI 开发，到做出一个带 RAG + Agent + 流式输出 + PDF 解析的完整问答系统，这个仓库记录了我完整的成长轨迹。

---

## 学习路径总览

这个项目经历了 **5 个版本、5 次认知升级**，每一次升级都是因为遇到了一个"真问题"：

```
v1 → 关键词搜索（我学会了：调API、写接口）
 │    ↘ 发现问题：搜"休假"找不到"年假"——关键词不靠谱
 ▼
v2 → 向量语义搜索（我学会了：embeddings、相似度计算）
 │    ↘ 发现问题：只能查固定知识库，功能太单一
 ▼
v3 → Agent 工具调用（我学会了：function calling、多工具路由）
 │    ↘ 发现问题：回答要等全部生成完才显示，体验差
 ▼
v4 → 流式输出（我学会了：SSE、流式响应）
 │    ↘ 发现问题：知识库是写死在代码里的，不够灵活
 ▼
v5 → PDF 知识库问答（我学会了：PDF解析、文件上传、完整产品闭环）
```

---

## 版本演进（为什么 + 做了什么）

### v1 — 从"调通 API"开始（入门）

**关键词：** `rag_api.py`

我第一次写 AI 应用时，目标是"能调通大模型 API 就行"。v1 的做法很简单：
- 把知识库文档写死在 Python 列表里
- 用户问问题 → 用关键词匹配找相关文档（搜"年假"才找得到"年假"）
- 把找到的文档塞进 Prompt → 调 DeepSeek API → 返回回答

**当时觉得好神奇，但很快发现了问题：** 搜"休假"找不到"年假"，搜"请假"找不到"调休"——关键词搜索太死板了。

---

### v2 — 让机器"理解意思"（升级）

**关键词：** `rag_api_v2.py`

为了解决搜不到同义词的问题，我学了**向量检索**：
- 用 Sentence-Transformers 把文档转成向量（一组数字）
- 用户提问也转成向量 → 算余弦相似度 → 找意思最接近的文档
- 同时换上了真正的 DeepSeek 模型回复（v1 用的是假回复）

**效果：** 搜"休假"也能找到"年假"了，因为它们的向量是接近的。**这是从"机械匹配"到"语义理解"的飞跃。**

---

### v3 — 给模型装上"工具箱"（进化）

**关键词：** `rag_agent.py`

v2 只能查知识库，我想让模型能干更多事——查天气、算数学。于是学了 **Agent（Function Calling）**：
- 给 DeepSeek 定义三个工具：`knowledge_base`（查知识库）、`get_weather`（查天气）、`calculate`（算数学）
- 模型自己判断用户想问什么 → 自动选工具 → 执行工具 → 用工具结果生成回答

**这是从"单一功能"到"多功能智能路由"的进化。**

---

### v4 — 像真人一样"逐字说话"（体验优化）

**关键词：** `rag_agent_stream.py`

前面的版本都是等全部回答生成完才一次性返回，用户体验不好。我学了**流式输出（SSE）**：
- 模型每生成一个字就立刻推送到前端
- 用户看到的是一个字一个字"冒出来"的效果
- 配套写了前端 `stream_test.html`，学了 Server-Sent Events 怎么对接

**这是从"能用"到"好用"的体验升级。**

---

### v5 — 做成真正的产品（完整闭环）

**关键词：** `pdf_qa.py`

前面的知识库都是写死在代码里的。我想让用户**自己上传 PDF，自动建库，直接问答**：
- 用户上传 PDF → 用 PyMuPDF 解析成文字 → 分块 → 转向量 → 存内存
- 然后就像 v2/v3 一样做 RAG 问答
- 结合了 v4 的流式输出，体验完整

**这是从"功能片段"到"完整产品"的最后一步。**

---

> 每一个版本我都写了对应的原理 demo 文件，方便后来者理解：
> - `vector_vs_keyword.py` — 向量搜索 vs 关键词搜索对比
> - `agent_demo.py` — Agent 功能调用是怎么工作的
> - `stream_demo.py` — 流式输出背后的机制
> - `pdf_parser_demo.py` — PDF 是怎么被解析成文字的

---

## 项目结构

```
rag-knowledge-base/
├── rag_api.py                # v1: 关键词搜索 RAG
├── rag_api_v2.py             # v2: 向量搜索 RAG
├── rag_agent.py              # v3: RAG + Agent 工具调用
├── rag_agent_stream.py       # v4: 流式输出版
├── pdf_qa.py                 # v5: PDF 知识库问答系统（集大成）
│
├── agent_demo.py             # Agent 基础原理 demo
├── stream_demo.py            # 流式输出原理 demo
├── vector_vs_keyword.py      # 向量 vs 关键词对比 demo
├── pdf_parser_demo.py        # PDF 解析分块 demo
├── pdf_pipeline_demo.py      # PDF 全流程可视化 demo
├── upload_demo.py            # 文件上传 demo
│
├── qa_test.html              # 前端测试页面
├── stream_test.html          # 流式测试页面
│
├── config.py                 # API Key 配置（不提交 Git）
├── config.example.py         # 配置模板
├── requirements.txt          # 依赖清单
├── .gitignore                # Git 忽略规则
├── AI学习笔记-完整总结.md     # 学习笔记
└── README.md                 # 项目说明
```

---

## 架构图（最终版 v5）

```
用户 POST /ask {"question": "年假怎么请？"}
        │
        ▼
┌──────────────────────────────────────────────┐
│  pdf_qa.py（FastAPI 服务端）                   │
│                                               │
│  ① Agent 判断 → 用哪个工具                    │
│  ② 向量检索 → 召回 Top-3 相关文档             │
│  ③ 文档 + 问题 拼接成 Prompt                  │
│  ④ 调 DeepSeek API 流式生成回答               │
│  ⑤ StreamingResponse 逐字返回                 │
└──────────────────────────────────────────────┘
        │
        ▼
      "根据公司制度，年假需提前3个工作日通过OA系统申请..."

（这个架构不是一开始就有的——v1 只有 ①②③④中的一小部分，
  每升一个版本才加上一块，v5 集大成。）
```

---

## 技术栈

| 层 | 技术 | 在哪个版本引入 |
|----|------|----------------|
| **后端框架** | Python + FastAPI | v1 |
| **大模型** | DeepSeek API（兼容 OpenAI 格式） | v1 |
| **关键词搜索** | Python 字符串匹配 | v1（v2 废弃） |
| **向量检索** | Sentence-Transformers（bge-small-zh-v1.5） | v2 |
| **向量计算** | NumPy 余弦相似度 | v2 |
| **Agent 工具调用** | DeepSeek Function Calling | v3 |
| **流式输出** | Server-Sent Events（SSE） | v4 |
| **PDF 解析** | PyMuPDF | v5 |
| **API 文档** | Swagger UI（自动生成） | v1 |

---

## 快速开始

### 前置条件

- Python 3.9+
- DeepSeek API Key（[注册](https://platform.deepseek.com/)）

### 安装

```bash
git clone https://github.com/ggtony81-svg/rag-knowledge-base.git
cd rag-knowledge-base

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install PyMuPDF python-multipart -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 配置

```bash
cp config.example.py config.py
# 编辑 config.py，填入你的 DEEPSEEK_API_KEY
```

### 启动

```bash
# PDF 知识库问答系统（推荐）
python pdf_qa.py

# 或 RAG + Agent + 流式版
python rag_agent_stream.py
```

访问 `http://127.0.0.1:18003/docs`（pdf_qa）或 `http://127.0.0.1:18001/docs`（rag_agent_stream）

### 前端测试

直接双击打开 `qa_test.html` 即可在浏览器中体验完整功能。

---

## API 接口

### POST /upload（pdf_qa）
上传 PDF 文件，自动解析并构建向量知识库。

### POST /ask/stream
提问（流式返回），模型基于知识库内容回答。

### POST /ask
提问（非流式），单次返回完整回答。

---

## License

MIT
