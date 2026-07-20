"""
PDF 问答 + RAG 优化版
优化1：语义分块 + 重叠上下文
优化2：CrossEncoder 重排序（先用向量搜 Top-10，再用 Rerank 精排 Top-3）
"""
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn, os, json, requests, numpy as np, pymysql

import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.remove(script_dir)
import redis as redis_lib
sys.path.insert(0, script_dir)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import DEEPSEEK_API_KEY

print("加载模型中...")
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
print("向量模型就绪")
reranker = CrossEncoder("BAAI/bge-reranker-base")
print("Rerank 模型就绪")

doc_chunks, doc_vectors = [], None
UPLOAD_DIR = "d:/shuqi/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
r = redis_lib.Redis(host='localhost', port=6379, decode_responses=True)
print("Redis 就绪")

def get_db():
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='pdf_qa', charset='utf8mb4')
    return conn, conn.cursor()

try:
    conn, cursor = get_db()
    cursor.execute("""CREATE TABLE IF NOT EXISTS conversations (id INT AUTO_INCREMENT PRIMARY KEY, title VARCHAR(255) NOT NULL DEFAULT '新对话', filename VARCHAR(255), created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP)""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS messages (id INT AUTO_INCREMENT PRIMARY KEY, conversation_id INT NOT NULL, role VARCHAR(20) NOT NULL, content TEXT NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE)""")
    conn.commit(); cursor.close(); conn.close()
    print("MySQL 就绪")
except Exception as e: print(f"MySQL: {e}")

def smart_chunk(text, overlap=50):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    chunks, cur = [], ""
    for line in lines:
        if any(line.startswith(p) for p in ["第一章","第二章","第三章","第一条","第二条","第三条","第四条","第五条","第六条","第七条"]):
            if cur: chunks.append(cur.strip()); cur = ""
        cur += line + "\n"
    if cur: chunks.append(cur.strip())
    res = []
    for i, c in enumerate(chunks):
        if i > 0: c = chunks[i-1][-overlap:] + "\n" + c
        res.append(c)
    return [c for c in res if len(c) > 15]

app = FastAPI(title="PDF 问答优化版")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
current_filename = "未命名"

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global doc_chunks, doc_vectors, current_filename
    current_filename = file.filename
    p = os.path.join(UPLOAD_DIR, file.filename)
    with open(p, "wb") as f: f.write(await file.read())
    import fitz
    doc = fitz.open(p); t = "".join(page.get_text() for page in doc)
    n = len(doc); doc.close()
    doc_chunks = smart_chunk(t)
    doc_vectors = model.encode(doc_chunks)
    return {"filename": file.filename, "pages": n, "chunks": len(doc_chunks), "total_chars": len(t)}

class ChatRequest(BaseModel):
    question: str
    conversation_id: int | None = None

@app.post("/ask/stream")
def ask_stream(req: ChatRequest):
    if not doc_chunks: return StreamingResponse(iter(["请先上传 PDF"]), media_type="text/plain")
    conv_id = req.conversation_id
    try:
        conn, cursor = get_db()
        if conv_id is None:
            cursor.execute("INSERT INTO conversations (title, filename) VALUES (%s, %s)", (req.question[:30], current_filename))
            conn.commit(); conv_id = cursor.lastrowid
        cursor.execute("INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)", (conv_id, "user", req.question))
        conn.commit(); cursor.close(); conn.close()
    except: pass
    cache_key = f"qa:{req.question}"
    cached = r.get(cache_key)
    if cached:
        try:
            conn, cursor = get_db()
            cursor.execute("INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)", (conv_id, "assistant", cached))
            conn.commit(); cursor.close(); conn.close()
        except: pass
        def fc(): yield cached
        return StreamingResponse(fc(), media_type="text/plain")

    # 两阶段检索
    qv = model.encode([req.question])
    # 阶段1: 向量粗搜 Top-10
    scores = [np.dot(qv[0], dv) / (np.linalg.norm(qv[0]) * np.linalg.norm(dv)) for dv in doc_vectors]
    top10 = np.argsort(scores)[-10:][::-1]
    # 阶段2: CrossEncoder 精排 Top-3
    pairs = [(req.question, doc_chunks[i]) for i in top10]
    re_scores = reranker.predict(pairs)
    top3 = [top10[i] for i in np.argsort(re_scores)[-3:][::-1]]
    results = [(doc_chunks[top10[i]], round(float(re_scores[i]), 2)) for i in np.argsort(re_scores)[-3:][::-1]]

    ctx = "\n".join([d for d, _ in results])

    full_answer = ""
    def generate():
        nonlocal full_answer
        resp = requests.post("https://api.deepseek.com/chat/completions", headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}, json={"model": "deepseek-chat", "messages": [{"role": "system", "content": f"基于以下资料回答问题：\n{ctx}"}, {"role": "user", "content": req.question}], "stream": True, "temperature": 0.7, "max_tokens": 2048}, stream=True, timeout=30)
        for line in resp.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    try:
                        c = json.loads(line[6:])["choices"][0].get("delta",{}).get("content","")
                        if c: full_answer += c; yield c
                    except: pass
        r.setex(cache_key, 3600, full_answer)
        try:
            conn, cursor = get_db()
            cursor.execute("INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)", (conv_id, "assistant", full_answer))
            cursor.execute("UPDATE conversations SET title=(SELECT content FROM messages WHERE conversation_id=%s AND role='user' ORDER BY id LIMIT 1) WHERE id=%s", (conv_id, conv_id))
            conn.commit(); cursor.close(); conn.close()
        except: pass
    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/conversations")
def list_convs():
    try:
        conn, cursor = get_db()
        cursor.execute("SELECT id, title, filename, updated_at FROM conversations ORDER BY updated_at DESC")
        rows = cursor.fetchall(); cursor.close(); conn.close()
        return {"conversations": [{"id": r[0], "title": r[1][:30], "filename": r[2], "time": str(r[3])} for r in rows]}
    except: return {"conversations": []}

@app.get("/conversation/{cid}")
def get_conv(cid: int):
    try:
        conn, cursor = get_db()
        cursor.execute("SELECT role, content FROM messages WHERE conversation_id=%s ORDER BY id", (cid,))
        rows = cursor.fetchall(); cursor.close(); conn.close()
        return {"messages": [{"role": r[0], "content": r[1]} for r in rows]}
    except: return {"messages": []}

@app.post("/conversation/new")
def new_conv():
    try:
        conn, cursor = get_db()
        cursor.execute("INSERT INTO conversations (title, filename) VALUES (%s, %s)", ("新对话", current_filename))
        conn.commit(); cid = cursor.lastrowid; cursor.close(); conn.close()
        return {"conversation_id": cid}
    except Exception as e: return {"error": str(e)}

@app.get("/")
def root():
    try:
        conn, cursor = get_db()
        cursor.execute("SELECT COUNT(*) FROM messages"); n = cursor.fetchone()[0]; cursor.close(); conn.close()
    except: n = 0
    return {"status": "ok", "chat_records": n, "cache_size": r.dbsize()}

if __name__ == "__main__":
    print("== PDF 问答优化版启动 ==")
    print("   CrossEncoder 重排序")
    print("== http://127.0.0.1:18005 ==")
    uvicorn.run(app, host="127.0.0.1", port=18005)
