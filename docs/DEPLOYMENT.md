# 系统部署说明

**项目名称**：基于检索增强的 PDF 知识库问答系统
**系统形态**：Web 应用（FastAPI 后端 + 原生 HTML/CSS/JS 前端）
**访问方式**：浏览器打开 `http://127.0.0.1:18005`
**日期**：2026-10-02

---

## 一、运行环境

### 1.1 硬件环境

本项目**在无独立显卡的普通笔记本上完成全部训练与评测**。

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows 11 |
| CPU | 6 物理核 / 12 逻辑核 |
| GPU | **无**（全部计算走 CPU） |
| 内存 | 建议 8GB 以上（模型加载约占 1GB） |

> 之所以强调"无 GPU"：向量模型微调在 CPU 上耗时 3.6 小时，这是本项目选择"小模型 + 子集调参"路线的原因，也是论文中多处实验设计的现实约束。

### 1.2 软件环境

实测通过的版本（`pip list` 可查）：

| 软件 | 版本 | 用途 |
|---|---|---|
| Python | 3.13.7 | 运行环境 |
| fastapi | 0.139.1 | 后端框架 |
| uvicorn | 0.51.0 | ASGI 服务器 |
| sentence-transformers | 5.6.0 | 向量模型加载与微调 |
| torch | 2.12.0 | 深度学习框架（CPU 版） |
| transformers | 5.14.1 | 底层模型库 |
| numpy | 2.4.4 | 向量运算 |
| pandas | 3.0.2 | 评测数据处理 |
| PyMuPDF | 1.28.0 | PDF 文本提取 |
| pymysql | 1.2.0 | MySQL 客户端 |
| redis | 4.6.0 | 缓存客户端 |
| jieba | 0.42.1 | BM25 中文分词（评测用） |
| matplotlib | 3.11.2 | 生成论文图表（评测用） |

> **注意**：`requirements.txt` 中列有 `datasets` 库，但经核查**代码中并未实际引用**（数据处理直接用 pandas + pyarrow 读 parquet）。不装不影响运行。

---

## 二、目录结构

```
rag-knowledge-base/
├── pdf_qa_optimized.py      # 线上主程序（FastAPI，端口 18005）
├── text_chunk.py            # 分块实现，线上与 eval/ 共用同一份
├── config.py                # 配置（含 API Key，**不随仓库分发**）
├── config.example.py        # 配置模板
├── requirements.txt
├── web/                     # 前端：原生 HTML/CSS/JS，无框架
│   ├── index.html
│   └── app.js
│
├── eval/                    # 离线评测体系
│   ├── data/                # 数据集（需下载，见第五节）
│   ├── cache/               # 微调后的模型权重（约 93MB，不随仓库分发）
│   ├── results/             # 评测结果、图表、日志
│   ├── download_data.py     # 下载 CMRC2018
│   ├── finetune_embedding.py    # 领域微调
│   ├── run_retrieval_eval.py    # 6 种检索方案对比
│   ├── run_e2e_eval.py          # 端到端问答评测
│   └── ...
│
├── uploads/                 # 用户上传的 PDF（运行时生成）
└── knowledge_base/          # 知识库落盘（运行时生成）
```

---

## 三、安装步骤

### 3.1 获取代码

```bash
git clone https://github.com/ggtony81-svg/rag-knowledge-base.git
cd rag-knowledge-base
```

### 3.2 安装依赖

国内建议走清华镜像：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 若只需运行线上服务、不做评测，可以只装前 10 个包（`fastapi` 到 `redis`），跳过 `datasets / pandas / pyarrow`。

### 3.3 配置 API Key

```bash
cp config.example.py config.py
```

编辑 `config.py`，填入 DeepSeek 的 API Key：

```python
DEEPSEEK_API_KEY = "sk-你的实际key"
```

> **`config.py` 含明文密钥，已在 `.gitignore` 中排除，切勿提交或打包分发。**

---

## 四、可选组件（MySQL 与 Redis）

**两者都是可选的。** 不装也能完整运行问答功能，系统会自动走降级分支。

| 组件 | 用途 | 不装的后果 |
|---|---|---|
| Redis | 问答结果缓存 | 每次提问都重新检索 + 调用大模型（实测 1214ms → 234ms） |
| MySQL | 保存会话与历史消息 | 不记录历史，每次问答独立 |

### 4.1 Redis

默认连接 `localhost:6379`。Windows 下可用免安装版：

```bash
cd redis
./redis-server.exe --port 6379 --save "" --appendonly no
```

验证是否就绪：

```bash
curl http://127.0.0.1:18005/status
# 返回 "cache_on": true 表示已连接
```

### 4.2 MySQL

需要先建库（表结构首次启动时自动创建）：

```sql
CREATE DATABASE pdf_qa DEFAULT CHARACTER SET utf8mb4;
```

连接参数写在 `pdf_qa_optimized.py` 的 `get_db()` 中，默认：

```python
host='localhost', user='root', password='123456', database='pdf_qa'
```

> 部署到其它机器时需按实际情况修改。**默认密码是弱口令，仅适用于本机演示环境。**

启动时会自动建立两张表：

- `conversations`（会话）
- `messages`（消息，外键指向 `conversations`，级联删除）

---

## 五、模型准备

线上默认加载**微调后**的模型（`eval/cache/finetuned_model`）。仓库不含模型权重（约 93MB），有两种方式：

### 方式一：先用通用模型跑起来（快）

```bash
MODEL_PATH=BAAI/bge-small-zh-v1.5 python pdf_qa_optimized.py
```

- 耗时：仅需下载模型（约 90MB）
- 效果：检索 P@1 **85.9%**
- 适用：只想先看看系统跑起来什么样

### 方式二：完整复现（慢，但指标正确）

```bash
python eval/download_data.py        # 下载 CMRC2018（约 5.7MB）
python eval/finetune_embedding.py   # 微调，CPU 上约 3.6 小时
python pdf_qa_optimized.py
```

- 效果：检索 P@1 **92.4%**
- 适用：复现论文中的评测结论

> **为什么不做"找不到微调模型就自动退回通用模型"的兜底？**
> 那样指标会从 92.4% 悄悄掉到 85.9%，界面上没有任何提示。
> **宁可启动报错，也不要静默降级。**

首次运行需联网下载基础模型，国内走 `hf-mirror.com` 镜像（已在代码中设置 `HF_ENDPOINT`）。

---

## 六、启动与访问

```bash
python pdf_qa_optimized.py
```

看到以下输出即表示启动成功：

```
加载 embedding 模型: .../eval/cache/finetuned_model
模型就绪
Redis 就绪
MySQL 就绪
Uvicorn running on http://127.0.0.1:18005
```

浏览器打开 **http://127.0.0.1:18005**

### 6.1 验证部署是否正常

```bash
curl http://127.0.0.1:18005/status
```

期望返回：

```json
{"status": "ok", "cache_size": 0, "cache_on": true,
 "kb": false, "file": null, "model": "finetuned_model"}
```

关键字段：
- `status: ok` —— 服务正常
- `model` —— **确认加载的是 `finetuned_model` 而不是通用模型**
- `kb: false` —— 尚未上传知识库，属正常

再看知识库与模型详情：

```bash
curl http://127.0.0.1:18005/kb/status
# {"loaded": false, "filename": null, "chunks": 0,
#  "model": "finetuned_model", "embed_dim": 512}
```

> `embed_dim` 应为 **512**，与 `bge-small-zh-v1.5` 一致。

### 6.2 使用流程

1. 点击上传区域，选择一个 **PDF 文件**（需含文字层，扫描件无法提取）
2. 系统自动解析 → 分块 → 向量化 → 落盘
3. 在输入框提问，回答**流式返回**，并展示检索到的原文片段

没有现成的 PDF 可以试？先生成一份测试文档：

```bash
python generate_test_pdf.py    # 生成 test_handbook.pdf（星海科技员工手册）
```

---

## 七、复现评测

```bash
# 检索方案对比（6 种方案）
python eval/run_retrieval_eval.py --schemes bm25 dense dense_rerank dense_ft dense_chunk dense_chunk_ft
python eval/summarize_results.py        # 汇总成表格

# 端到端问答评测（200 条，约 10 分钟）
python eval/run_e2e_eval.py

# 黑盒功能测试（需要服务已启动）
python eval/run_blackbox_tests.py

# 生成论文图表
python eval/make_figures.py
```

> **黑盒测试建议在 Redis 开、停两种状态下各跑一次**。只测"依赖都正常"会漏掉降级路径的问题——本项目就曾因此在 `/status` 接口上遗留过一个 Redis 停止即返回 500 的缺陷。

---

## 八、常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 启动报错找不到模型 | 未做微调，且未指定通用模型 | 用方式一启动，或先跑 `finetune_embedding.py` |
| 上传 PDF 提示"没有提取到可用文字" | 扫描件/纯图片，无文字层 | 换一份有文字层的 PDF，或先做 OCR |
| 上传提示"不是有效的 PDF" | 文件损坏，或扩展名被改成 `.pdf` | 确认文件本身是有效 PDF |
| `/status` 返回 `cache_on: false` | Redis 未启动 | 正常降级，缓存不可用但功能不受影响；需缓存则启动 Redis |
| `MySQL: ...` 启动时打印报错 | MySQL 未启动或密码不符 | 正常降级，不影响问答；需历史记录则修正 `get_db()` 配置 |
| 回答速度慢（约 10 秒） | 首次问答需调用大模型生成 | 属正常；重复提问会命中缓存（约 0.2 秒） |
| 端口 18005 被占用 | 已有实例在运行 | 关闭旧进程，或修改 `uvicorn.run()` 中的端口 |

---

## 九、部署注意事项

### 9.1 目录路径

`pdf_qa_optimized.py` 中的知识库目录与上传目录**相对脚本所在目录**解析，不依赖部署位置：

```python
KB_DIR = os.path.join(script_dir, "knowledge_base")
UPLOAD_DIR = os.path.join(script_dir, "uploads")
```

`generate_test_pdf.py` 的默认输出路径同理。

> 早期版本这两处写死的是 `d:/shuqi/...`，换台机器 clone 下来会往错误的位置写（或者直接失败）。
> 已改为相对路径 —— 因为脚本本来就在项目根目录下，本机行为不变，但换机器不再需要手改代码。

### 9.2 本项目为单机演示环境

以下问题在正式部署前需要处理，本项目未做：

| 问题 | 说明 |
|---|---|
| 无身份认证 | 任何能访问该端口的人都能上传文件、提问 |
| 无 HTTPS | 明文传输 |
| 数据库中存弱口令 | 默认 `root/123456` |
| 未做并发压测 | 全局知识库状态（`doc_chunks` 等）在并发上传时会存在竞争，实测未验证 |
| 未做上传大小限制 | 超大 PDF 可能耗尽内存 |

论文第八章"存在的不足"中已如实说明这些局限。

### 9.3 提交与分发时需排除的文件

打包源代码时**必须排除**：

| 文件/目录 | 原因 |
|---|---|
| `config.py` | **含明文 API Key** |
| `__pycache__/` | 其中的 `config.cpython-313.pyc` 同样含 Key |
| `eval/cache/` | 模型权重，约 93MB |
| `uploads/`、`knowledge_base/` | 运行时生成的用户数据 |
| `redis/` | 第三方二进制 |

前两项已在 `.gitignore` 中，但**打包压缩不会自动应用 gitignore 规则**，需手工排除。
