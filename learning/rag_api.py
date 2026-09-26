"""
把 RAG 知识库包装成 API 接口
别人可以通过 HTTP 调你的知识库问答系统
"""

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import requests
import json
from config import DEEPSEEK_API_KEY

# ============================================================
# 第1部分：建立知识库（模拟）
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

def call_deepseek(messages):
    """调 DeepSeek 官方接口"""
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
        result = resp.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"（调用大模型出错：{e}）"

# 模拟向量搜索：用关键词粗略检索相关文档
def search_docs(query):
    # 真实项目中这里调向量数据库
    # 这里用关键词模拟：找到包含问题里任意词的文档
    keywords = query.replace("?", "").replace("？", "").replace("的", "").replace("了", "")
    results = []
    for doc in documents:
        score = sum(1 for kw in keywords if kw in doc)
        if score > 0:
            results.append((doc, score))
    # 按相关度排序
    results.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in results[:2]]

# 模拟调大模型
def fake_llm_reply(context, question):
    if "请假" in question or "旅游" in question:
        return "根据公司制度，您可以使用年假或事假。年假需提前3个工作日申请，事假全年累计不超过15天。"
    elif "工资" in question:
        return "薪资每月15日发放，如遇节假日会提前发放。"
    elif "加班" in question:
        return "加班可以申请调休或按1.5倍工资计算加班费。"
    else:
        return f"根据查询到的资料：{context[:30]}... 请补充您的问题。"


# ============================================================
# 第2部分：定义 API
# ============================================================
app = FastAPI(title="公司知识库 RAG 系统")

# 定义请求的数据格式
class ChatRequest(BaseModel):
    question: str                       # 用户的问题

# 定义响应的数据格式
class ChatResponse(BaseModel):
    question: str                       # 原问题
    answer: str                         # 回答
    source_docs: list[str]              # 参考了哪些资料

@app.post("/ask", response_model=ChatResponse)
def ask_question(req: ChatRequest):
    """
    用户发 POST 请求到 /ask
    数据格式：{"question": "年假怎么请？"}
    返回格式：{"question": "...", "answer": "...", "source_docs": [...]}
    """
    # 第1步：搜索相关文档（模拟向量检索）
    relevant_docs = search_docs(req.question)

    # 第2步：拼接上下文
    context = "\n".join(relevant_docs)

    # 第3步：拼接 prompt 调真实的 DeepSeek
    messages = [
        {"role": "system", "content": f"你是一个公司知识库助手。请基于以下资料回答问题。如果资料里没有相关信息，就说不知道，不要编造。\n\n资料：\n{context}"},
        {"role": "user", "content": req.question}
    ]
    answer = call_deepseek(messages)

    # 第4步：返回结果
    return ChatResponse(
        question=req.question,
        answer=answer,
        source_docs=relevant_docs
    )

@app.get("/")
def root():
    """根路径，确认服务是否运行"""
    return {
        "service": "公司知识库 RAG 系统",
        "status": "running",
        "usage": "POST /ask  提问格式：{\"question\": \"你的问题\"}"
    }

# ============================================================
# 第3部分：启动服务
# ============================================================
if __name__ == "__main__":
    print("🚀 RAG 知识库 API 已启动！")
    print("📝 接口地址：http://127.0.0.1:18001")
    print("📝 提问接口：POST http://127.0.0.1:18001/ask")
    print("📝 请求示例：{\"question\": \"年假怎么请？\"}")
    print("-" * 50)
    uvicorn.run(app, host="127.0.0.1", port=18001)
