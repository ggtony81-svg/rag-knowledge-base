"""
RAG API v2 — 用真正的向量搜索替代关键词匹配
"""
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import requests
from config import DEEPSEEK_API_KEY

# ============================================================
# 第1部分：用向量数据库建知识库
# ============================================================
documents = [
    "入职满一年员工可享受5天年假，满10年可享受10天年假",
    "年假需提前3个工作日通过OA系统申请",
    "春节放假7天，国庆放假7天，元旦放假3天",
    "病假需提供二级以上医院开具的证明",
    "事假全年累计不得超过15天，需部门主管审批",
    "薪资每月15日发放，遇节假日提前",
    "加班可申请调休或按1.5倍工资计算加班费",
]

print("⏳ 加载 Embedding 模型中...")

# 设置国内镜像（HuggingFace 被墙了）
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer
import numpy as np

# 加载 embedding 模型（把文字转成向量）
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
print("✅ Embedding 模型加载完成")

# 把所有文档转成向量（建索引）
print("⏳ 正在构建向量索引...")
doc_vectors = model.encode(documents)
print(f"✅ {len(documents)} 条文档 → {len(doc_vectors[0])} 维向量")

def vector_search(query, top_k=2):
    """向量搜索：把问题转向量 → 算相似度 → 返回最相关的文档"""
    q_vec = model.encode([query])
    # 算余弦相似度
    scores = []
    for d_vec in doc_vectors:
        cos_sim = np.dot(q_vec[0], d_vec) / (np.linalg.norm(q_vec[0]) * np.linalg.norm(d_vec))
        scores.append(cos_sim)
    # 取 top_k
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [(documents[i], scores[i]) for i in top_indices]

# ============================================================
# 第2部分：调 DeepSeek API
# ============================================================
def call_deepseek(messages):
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024
    }
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"（调用大模型出错：{e}）"

# ============================================================
# 第3部分：定义 API 接口
# ============================================================
app = FastAPI(title="公司知识库 RAG 系统 v2 — 向量搜索版")

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question: str
    answer: str
    source_docs: list[dict]   # 改为返回文档 + 相似度分数

@app.post("/ask", response_model=ChatResponse)
def ask_question(req: ChatRequest):
    # 向量搜索
    results = vector_search(req.question, top_k=2)

    context = "\n".join([doc for doc, _ in results])

    messages = [
        {"role": "system", "content": f"你是一个公司知识库助手。请基于以下资料回答问题。如果资料里没有相关信息，就说不知道，不要编造。\n\n资料：\n{context}"},
        {"role": "user", "content": req.question}
    ]

    answer = call_deepseek(messages)

    return ChatResponse(
        question=req.question,
        answer=answer,
        source_docs=[{"doc": doc, "similarity": round(float(score), 2)} for doc, score in results]
    )

@app.get("/")
def root():
    return {
        "service": "公司知识库 RAG 系统 v2",
        "status": "running",
        "version": "向量搜索版",
        "usage": "POST /ask  提问格式：{\"question\": \"你的问题\"}"
    }

# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    print("🚀 RAG v2 启动（向量搜索版）")
    print("📝 接口地址：http://127.0.0.1:18001")
    print("📝 API 文档：http://127.0.0.1:18001/docs")
    print("-" * 50)
    uvicorn.run(app, host="127.0.0.1", port=18001)
