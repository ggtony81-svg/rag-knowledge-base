"""
RAG + Agent + 流式输出 融合版
"""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn, requests, json, numpy as np, os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
from config import DEEPSEEK_API_KEY

# ============================================================
# 知识库
# ============================================================
print("加载 Embedding 模型中...")
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
print(f"OK 知识库加载完成：{len(documents)} 条文档")

def search_knowledge_base(query):
    q_vec = model.encode([query])
    scores = [np.dot(q_vec[0], d_vec) / (np.linalg.norm(q_vec[0]) * np.linalg.norm(d_vec)) for d_vec in doc_vectors]
    top_idx = np.argsort(scores)[-2:][::-1]
    return [documents[i] for i in top_idx]

# ============================================================
# 工具
# ============================================================
def get_weather(city):
    data = {"北京": "25°C晴转多云", "上海": "28°C有雨", "广州": "32°C闷热", "深圳": "30°C阵雨"}
    return data.get(city, f"{city}暂无数据")

def calculate(expr):
    try:
        safe = "".join(c for c in expr if c in "0123456789+-*/(). ")
        return f"{expr} = {eval(safe)}"
    except:
        return f"无法计算：{expr}"

tools = [
    {"type": "function", "function": {"name": "knowledge_base", "description": "搜索公司知识库",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_weather", "description": "查询天气",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "calculate", "description": "计算数学",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}
]

# ============================================================
# Agent 逻辑（非流式，跑工具用）
# ============================================================
def call_deepseek(messages):
    resp = requests.post("https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
        json={"model": "deepseek-chat", "messages": messages, "tools": tools, "tool_choice": "auto",
              "temperature": 0.7, "max_tokens": 2048}, timeout=30)
    return resp.json()

def run_tools(user_input):
    """运行 Agent 工具调用阶段，只跑工具不生成回答"""
    messages = [
        {"role": "system", "content": "你是一个智能助手，可以使用工具。公司问题用knowledge_base，天气用get_weather，数学用calculate。"},
        {"role": "user", "content": user_input}
    ]
    for _ in range(5):
        response = call_deepseek(messages)
        msg = response["choices"][0]["message"]
        if not msg.get("tool_calls"):
            # 没有工具调用 → 不需要跑工具，直接返回给流式生成
            return messages
        messages.append(msg)
        for tc in msg["tool_calls"]:
            args = json.loads(tc["function"]["arguments"])
            if tc["function"]["name"] == "knowledge_base":
                result = json.dumps(search_knowledge_base(args["query"]), ensure_ascii=False)
            elif tc["function"]["name"] == "get_weather":
                result = get_weather(**args)
            elif tc["function"]["name"] == "calculate":
                result = calculate(**args)
            else:
                result = "未知工具"
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
    return messages

# ============================================================
# 流式生成
# ============================================================
def stream_generate(messages):
    """把消息发给 DeepSeek，流式逐字返回"""
    resp = requests.post("https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
        json={"model": "deepseek-chat", "messages": messages, "stream": True,
              "temperature": 0.7, "max_tokens": 2048}, stream=True, timeout=30)

    for line in resp.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    content = chunk["choices"][0].get("delta", {}).get("content", "")
                    if content:
                        yield content
                except:
                    pass

# ============================================================
# FastAPI
# ============================================================
app = FastAPI(title="RAG + Agent + 流式输出")

# 允许跨域（浏览器测试用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

@app.post("/ask/stream")
def ask_stream(req: ChatRequest):
    """流式接口：先跑 Agent 工具，再流式生成回答"""

    # 阶段1：跑 Agent 工具（不流式，速度很快）
    final_messages = run_tools(req.question)

    # 阶段2：流式生成回答
    return StreamingResponse(stream_generate(final_messages), media_type="text/plain")

@app.post("/ask")
def ask(req: ChatRequest):
    """普通接口（非流式），保留兼容"""
    messages = run_tools(req.question)
    # 把最后一条 assistant 消息的 content 取出来
    last_msg = messages[-1]
    if last_msg["role"] == "assistant":
        return {"answer": last_msg.get("content", "")}
    # 否则重新调一次非流式
    resp = requests.post("https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
        json={"model": "deepseek-chat", "messages": messages, "temperature": 0.7, "max_tokens": 2048}, timeout=30)
    return {"answer": resp.json()["choices"][0]["message"]["content"]}

@app.get("/")
def root():
    return {"service": "RAG + Agent + 流式输出", "usage": "POST /ask/stream 流式接口 | POST /ask 普通接口"}

if __name__ == "__main__":
    print("== RAG + Agent + 流式输出启动 ==")
    print("== http://127.0.0.1:18001/docs ==")
    uvicorn.run(app, host="127.0.0.1", port=18001)
