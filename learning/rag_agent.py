"""
RAG + Agent 融合版
用户提问 → Agent 决定用哪个工具 → 执行 → 回答

工具列表：
1. knowledge_base → 搜索公司知识库（向量检索）
2. get_weather   → 查天气
3. calculate     → 算数学
"""
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import requests
import json
import numpy as np
from config import DEEPSEEK_API_KEY

# ============================================================
# 第1部分：知识库（向量检索）
# ============================================================
print("[..] 加载 Embedding 模型中...")
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer

documents = [
    "入职满一年员工可享受5天年假，满10年可享受10天年假",
    "年假需提前3个工作日通过OA系统申请",
    "春节放假7天，国庆放假7天，元旦放假3天",
    "病假需提供二级以上医院开具的证明",
    "事假全年累计不得超过15天，需部门主管审批",
    "薪资每月15日发放，遇节假日提前",
    "加班可申请调休或按1.5倍工资计算加班费",
]

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
doc_vectors = model.encode(documents)
print(f"[OK] 知识库加载完成：{len(documents)} 条文档")

def search_knowledge_base(query):
    """向量搜索知识库"""
    q_vec = model.encode([query])
    scores = []
    for d_vec in doc_vectors:
        cos_sim = np.dot(q_vec[0], d_vec) / (np.linalg.norm(q_vec[0]) * np.linalg.norm(d_vec))
        scores.append(cos_sim)
    top_idx = np.argsort(scores)[-2:][::-1]
    results = [{"doc": documents[i], "similarity": round(float(scores[i]), 2)} for i in top_idx]
    return results

# ============================================================
# 第2部分：工具定义
# ============================================================

def get_weather(city):
    weather_data = {
        "北京": "25°C，晴转多云",
        "上海": "28°C，有雨",
        "广州": "32°C，闷热",
        "深圳": "30°C，阵雨",
    }
    return weather_data.get(city, f"{city}：暂时没有天气数据")

def calculate(expression):
    try:
        allowed = "0123456789+-*/(). "
        safe_expr = "".join(c for c in expression if c in allowed)
        result = eval(safe_expr)
        return f"{expression} = {result}"
    except:
        return f"无法计算：{expression}"

# 工具的说明书（发给 DeepSeek）
tools = [
    {
        "type": "function",
        "function": {
            "name": "knowledge_base",
            "description": "搜索公司知识库，查询公司制度、年假、薪资、考勤、福利等信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户的问题原文"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 1+2*3"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# ============================================================
# 第3部分：Agent 核心 — 调用模型 + 执行工具
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
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.7,
        "max_tokens": 2048
    }
    resp = requests.post(url, headers=headers, json=data, timeout=30)
    return resp.json()

def run_agent(user_input):
    """Agent 主循环：支持多轮工具调用"""
    messages = [
        {"role": "system", "content": "你是一个智能助手，可以使用工具来回答用户的问题。"
                                      "对于公司制度相关的问题，请使用 knowledge_base 工具搜索。"
                                      "如果工具返回的结果不足以回答，可以多次调用不同工具。"},
        {"role": "user", "content": user_input}
    ]

    # 最多允许 5 轮工具调用，防止死循环
    max_rounds = 5
    for round_num in range(max_rounds):
        response = call_deepseek(messages)
        msg = response["choices"][0]["message"]

        # 如果模型不再需要调工具，返回最终回答
        if not msg.get("tool_calls"):
            return msg["content"]

        # 把模型的工具调用指令加到对话里
        messages.append(msg)

        # 执行每个工具
        for tool_call in msg["tool_calls"]:
            func_name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])

            # 根据工具名执行不同的函数
            if func_name == "knowledge_base":
                result = json.dumps(search_knowledge_base(args["query"]), ensure_ascii=False)
            elif func_name == "get_weather":
                result = get_weather(**args)
            elif func_name == "calculate":
                result = calculate(**args)
            else:
                result = "未知工具"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result
            })

    return "（Agent 执行轮次过多，已终止）"

# ============================================================
# 第4部分：FastAPI 接口
# ============================================================

app = FastAPI(title="RAG + Agent 融合系统")

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question: str
    answer: str

@app.post("/ask", response_model=ChatResponse)
def ask(req: ChatRequest):
    answer = run_agent(req.question)
    return ChatResponse(question=req.question, answer=answer)

@app.get("/")
def root():
    return {
        "service": "RAG + Agent 融合系统",
        "status": "running",
        "tools": ["知识库搜索", "天气查询", "数学计算"]
    }

# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    print("== RAG + Agent 融合系统启动 ==")
    print("== http://127.0.0.1:18001/docs ==")
    uvicorn.run(app, host="127.0.0.1", port=18001)
