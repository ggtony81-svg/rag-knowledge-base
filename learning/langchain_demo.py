"""
LangChain 演示：跟手写 RAG 做对比
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ============================================================
# 你的手写 RAG（简化版）
# ============================================================
print("=" * 60)
print("你的手写 RAG（你之前写的代码逻辑）")
print("=" * 60)

sentences = [
    "入职满一年员工可享受5天年假",
    "年假需提前3个工作日申请",
    "加班按1.5倍工资计算加班费",
]

# 手写：自己转向量、自己算相似度
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
vecs = model.encode(sentences)

def handwrite_search(query):
    qv = model.encode([query])
    scores = [np.dot(qv[0], v) / (np.linalg.norm(qv[0]) * np.linalg.norm(v)) for v in vecs]
    top = np.argmax(scores)
    return sentences[top]

print(f"手写检索「加班怎么算」→ {handwrite_search('加班怎么算')}")
print()

# ============================================================
# LangChain 版本
# ============================================================
print("=" * 60)
print("LangChain 版本（相同的功能，更少的代码）")
print("=" * 60)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

# 1. 文档（自动拆分）
docs = [Document(page_content=s) for s in sentences]

# 2. Embedding 模型（一行搞定）
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5"
)

# 3. 向量库（自动存储 + 检索）
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 4. 检索
result = retriever.invoke("加班怎么算")
print(f"LangChain 检索「加班怎么算」→ {result[0].page_content}")
print()

# ============================================================
# 核心区别
# ============================================================
print("=" * 60)
print("手写 vs LangChain")
print("=" * 60)
print("""
手写 RAG（你的代码）:
  model.encode()         ← 自己转向量
  np.dot() / cosine      ← 自己算相似度
  np.argsort()           ← 自己排序
  print()                ← 自己展示结果

LangChain:
  embeddings = HuggingFaceEmbeddings()  ← 一行搞定
  vectorstore = Chroma.from_documents() ← 自动建库
  retriever.invoke()                    ← 自动检索

核心价值：
  LangChain 把 RAG 流程做成了"搭积木"
  你想换 embedding 模型？改一行参数
  你想换向量数据库？改一行
  你想加重排序？加一行 retriever = ...
""")

# 清理 Chroma 临时目录
import shutil
if os.path.exists("./chroma_data"):
    shutil.rmtree("./chroma_data")
