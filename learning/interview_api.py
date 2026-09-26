"""
AI 面试官后端 API
RAG 面试知识库 + Agent 工具 + 流式问答 + MySQL 历史
"""
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn, os, json, requests, numpy as np, pymysql, pickle, fitz

import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.remove(script_dir)
import redis as redis_lib
sys.path.insert(0, script_dir)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
from config import DEEPSEEK_API_KEY

print("加载模型中...")
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
print("模型就绪")

KB_DIR = "d:/shuqi/knowledge_base"
os.makedirs(KB_DIR, exist_ok=True)
r = redis_lib.Redis(host="localhost", port=6379, decode_responses=True)
print("Redis 就绪")

# ============================================================
# MySQL
# ============================================================
def get_db():
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='pdf_qa', charset='utf8mb4')
    return conn, conn.cursor()

try:
    conn, cursor = get_db()
    cursor.execute("""CREATE TABLE IF NOT EXISTS interviews (
        id INT AUTO_INCREMENT PRIMARY KEY,
        title VARCHAR(255), job_desc TEXT, resume TEXT,
        status VARCHAR(20) DEFAULT 'active',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS interview_msgs (
        id INT AUTO_INCREMENT PRIMARY KEY,
        interview_id INT NOT NULL,
        role VARCHAR(20), content TEXT, score INT DEFAULT NULL,
        comment TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (interview_id) REFERENCES interviews(id) ON DELETE CASCADE)""")
    conn.commit(); cursor.close(); conn.close()
    print("MySQL 就绪")
except Exception as e: print(f"MySQL: {e}")

# ============================================================
# 面试知识库（内置）
# ============================================================
kb_docs = [
    "自我介绍：控制在1-2分钟，包含基本信息、核心能力、亮点项目、求职动机，结构清晰不背稿。",
    "STAR法则：情境(Situation)→任务(Task)→行动(Action)→结果(Result)，回答行为类问题必备。",
    "为什么离职：客观原因为主，谈成长空间/职业规划，绝不说前公司坏话。",
    "期望薪资：先了解市场行情，给出区间，强调可谈，避免直接报死数。",
    "缺点问题：说真实的非致命缺点，并说明改进方法，避免'我太追求完美'这类套路。",
    "项目介绍：选1-2个最亮眼项目，讲清楚背景、我的角色、技术难点、量化成果。",
    "反问环节：问团队情况、技术栈、业务方向、晋升机制，展现积极性。",
    "Python基础：解释器/编译器的区别、GIL、装饰器原理、生成器与迭代器。",
    "FastAPI：依赖注入、Pydantic校验、异步支持、自动文档生成。",
    "MySQL：索引原理（B+树）、事务ACID、隔离级别、慢查询优化、主从复制。",
    "Redis：数据结构（string/hash/list/set/zset）、缓存穿透/击穿/雪崩、持久化RDB/AOF。",
    "RAG：文档分块→向量化→检索→重排序→生成，解决幻觉问题，企业落地主流方案。",
    "大模型API：流式输出原理（SSE）、temperature参数、token计算、上下文窗口。",
    "Docker：镜像/容器/卷/网络，Dockerfile编写，docker-compose多服务编排。",
    "HTTP：GET/POST区别、状态码含义、RESTful设计规范、JWT认证原理。",
    "设计模式：单例、工厂、策略、观察者，讲清适用场景。",
    "算法题：数组/链表/树/哈希表/动态规划，时间空间复杂度分析。",
    "AI项目：讲RAG知识库或Agent项目时，突出'为什么这么设计'和'踩过的坑'。",
    "团队协作：代码评审、Git分支管理、敏捷开发、沟通技巧。",
    "职业规划：3-5年技术深度为主，管理为辅，表达学习意愿和稳定性。",
]

def search_kb(query, top_k=3):
    """向量检索面试知识库"""
    qv = model.encode([query])
    scores = [np.dot(qv[0], model.encode([d])[0]) / (np.linalg.norm(qv[0]) * np.linalg.norm(model.encode([d])[0]))
              for d in kb_docs]
    top = np.argsort(scores)[-top_k:][::-1]
    return [kb_docs[i] for i in top]

app = FastAPI(title="AI 面试官")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ============================================================
# 工具：面试知识库检索
# ============================================================
tools = [{
    "type": "function",
    "function": {
        "name": "search_kb",
        "description": "检索面试知识库，获取常见面试问题的最佳回答思路",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "面试主题，如：自我介绍、项目介绍、Python基础"}
        }, "required": ["query"]}
    }
}]

def call_deepseek(messages, stream=False):
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "deepseek-chat", "messages": messages, "temperature": 0.7, "max_tokens": 2048, "stream": stream}
    if stream:
        return requests.post(url, headers=headers, json=data, stream=True, timeout=30)
    resp = requests.post(url, headers=headers, json=data, timeout=30)
    return resp.json()

# ============================================================
# 接口
# ============================================================
class SetupRequest(BaseModel):
    job_desc: str
    resume: str

class AskRequest(BaseModel):
    interview_id: int
    answer: str

@app.post("/interview/setup")
def setup_interview(req: SetupRequest):
    """创建面试：分析 JD + 简历，生成面试官设定"""
    try:
        conn, cursor = get_db()
        cursor.execute("INSERT INTO interviews (title, job_desc, resume) VALUES (%s, %s, %s)",
            (req.job_desc[:30], req.job_desc, req.resume))
        conn.commit()
        iid = cursor.lastrowid
        cursor.close(); conn.close()
    except: iid = 1

    prompt = f"""你是资深面试官。根据岗位要求和候选人简历，生成面试官设定。

岗位要求：
{req.job_desc}

候选人简历：
{req.resume}

请输出：
## 岗位核心要求
(列3-5个最关键的考察点)

## 面试策略
(首轮问什么、重点考察什么、追问方向)

## 开场白
(一句自然的开场，约50字)"""
    result = call_deepseek([{"role": "system", "content": "你是资深面试官。"}, {"role": "user", "content": prompt}])
    content = result["choices"][0]["message"]["content"]

    # 存开场白
    try:
        conn, cursor = get_db()
        cursor.execute("INSERT INTO interview_msgs (interview_id, role, content) VALUES (%s, %s, %s)",
            (iid, "assistant", content))
        conn.commit(); cursor.close(); conn.close()
    except: pass

    return {"interview_id": iid, "setup": content}

@app.post("/interview/ask")
def interview_ask(req: AskRequest):
    """候选人回答后，面试官追问 + 点评"""
    # 存回答
    try:
        conn, cursor = get_db()
        cursor.execute("INSERT INTO interview_msgs (interview_id, role, content) VALUES (%s, %s, %s)",
            (req.interview_id, "user", req.answer))
        conn.commit()
        cursor.execute("SELECT role, content FROM interview_msgs WHERE interview_id=%s ORDER BY id", (req.interview_id,))
        history = cursor.fetchall()
        cursor.close(); conn.close()
    except: history = []

    # 检索面试知识库
    kb_context = search_kb(req.answer)
    kb_text = "\n".join(kb_context)

    msgs = [{"role": "system", "content": f"""你是资深面试官。你正在面试候选人。

参考面试知识库（用于评价回答质量）：
{kb_text}

规则：
1. 先点评刚才的回答（评分+简短评语）
2. 再问下一个问题（可以追问或换话题）
3. 点评要具体、犀利，指出亮点和不足
4. 评分 1-10 分"""}]
    for row in history:
        msgs.append({"role": row[0], "content": row[1]})
    msgs.append({"role": "user", "content": "请点评我的回答并问下一个问题。"})

    def gen():
        resp = call_deepseek(msgs, stream=True)
        full = ""
        for line in resp.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    try:
                        c = json.loads(line[6:])["choices"][0].get("delta",{}).get("content","")
                        if c: full += c; yield c
                    except: pass
        # 存面试官回复
        try:
            conn, cursor = get_db()
            cursor.execute("INSERT INTO interview_msgs (interview_id, role, content) VALUES (%s, %s, %s)",
                (req.interview_id, "assistant", full))
            conn.commit(); cursor.close(); conn.close()
        except: pass
        # 缓存
        r.setex(f"interview:{req.interview_id}:last", 3600, full)
    return StreamingResponse(gen(), media_type="text/plain")

@app.get("/interview/history")
def list_interviews():
    try:
        conn, cursor = get_db()
        cursor.execute("SELECT id, title, status, created_at FROM interviews ORDER BY id DESC")
        rows = cursor.fetchall(); cursor.close(); conn.close()
        return {"interviews": [{"id": r[0], "title": r[1], "status": r[2], "time": str(r[3])} for r in rows]}
    except: return {"interviews": []}

@app.get("/interview/{iid}/messages")
def get_messages(iid: int):
    try:
        conn, cursor = get_db()
        cursor.execute("SELECT role, content FROM interview_msgs WHERE interview_id=%s ORDER BY id", (iid,))
        rows = cursor.fetchall(); cursor.close(); conn.close()
        return {"messages": [{"role": r[0], "content": r[1]} for r in rows]}
    except: return {"messages": []}

@app.delete("/interview/{iid}")
def delete_interview(iid: int):
    try:
        conn, cursor = get_db()
        cursor.execute("DELETE FROM interview_msgs WHERE interview_id=%s", (iid,))
        cursor.execute("DELETE FROM interviews WHERE id=%s", (iid,))
        conn.commit(); cursor.close(); conn.close()
        return {"status": "ok"}
    except: return {"error": "failed"}

@app.get("/")
def root():
    return {"service": "AI 面试官", "status": "running"}

if __name__ == "__main__":
    print("== AI 面试官启动 ==")
    print("== http://127.0.0.1:18006 ==")
    uvicorn.run(app, host="127.0.0.1", port=18006)
