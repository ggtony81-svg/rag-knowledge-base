"""
PDF 知识库问答系统 + MySQL 聊天记录存储
"""
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn, os, json, requests, numpy as np, pymysql
from datetime import datetime

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
from config import DEEPSEEK_API_KEY

print("加载模型中...")
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

doc_chunks = []
doc_vectors = None
UPLOAD_DIR = "d:/shuqi/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============================================================
# MySQL 连接
# ============================================================
def get_db():
    conn = pymysql.connect(host='localhost', user='root', password='123456',
                           database='pdf_qa', charset='utf8mb4')
    return conn, conn.cursor()

# 初始化表
try:
    conn, cursor = get_db()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            filename VARCHAR(255),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()
    print("MySQL 聊天记录表就绪 ✅")
except Exception as e:
    print(f"MySQL 连接失败: {e}")

# ============================================================
# 工具
# ============================================================
def chunk_text(text, chunk_size=200):
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    for sep in ["\n\n", "\n", "。", "；", "，"]:
        if sep in text:
            parts = text.split(sep)
            current = ""
            for part in parts:
                part = part.strip()
                if not part: continue
                if len(current) + len(part) < chunk_size:
                    current += part + sep
                else:
                    if current: chunks.append(current.strip())
                    current = part + sep
            if current: chunks.append(current.strip())
            break
    else:
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
    return [c for c in chunks if len(c) > 10]

# ============================================================
# API
# ============================================================
app = FastAPI(title="PDF 知识库问答系统（带 MySQL）")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

current_filename = "未命名"

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global doc_chunks, doc_vectors, current_filename
    current_filename = file.filename

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    import fitz
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    num_pages = len(doc)
    doc.close()

    doc_chunks = chunk_text(full_text)
    doc_vectors = model.encode(doc_chunks)

    return {
        "filename": file.filename,
        "pages": num_pages,
        "chunks": len(doc_chunks),
        "total_chars": len(full_text),
        "status": "知识库构建完成"
    }

class ChatRequest(BaseModel):
    question: str

@app.post("/ask/stream")
def ask_stream(req: ChatRequest):
    if not doc_chunks:
        return StreamingResponse(iter(["请先上传 PDF 文档"]), media_type="text/plain")

    # 向量检索
    q_vec = model.encode([req.question])
    scores = [np.dot(q_vec[0], d_vec) / (np.linalg.norm(q_vec[0]) * np.linalg.norm(d_vec))
              for d_vec in doc_vectors]
    top_idx = np.argsort(scores)[-3:][::-1]
    context = "\n".join([doc_chunks[i] for i in top_idx])

    full_answer = ""

    def generate():
        nonlocal full_answer
        resp = requests.post("https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": f"基于以下资料回答问题，不知道就说不知道：\n{context}"},
                    {"role": "user", "content": req.question}
                ],
                "stream": True, "temperature": 0.7, "max_tokens": 2048
            }, stream=True, timeout=30)

        for line in resp.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        content = chunk["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            full_answer += content
                            yield content
                    except:
                        pass

        # 流式完成后，把问答存进 MySQL
        try:
            conn, cursor = get_db()
            cursor.execute(
                "INSERT INTO chat_history (question, answer, filename) VALUES (%s, %s, %s)",
                (req.question, full_answer, current_filename)
            )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"存 MySQL 失败: {e}")

    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/history")
def get_history():
    """查看聊天历史"""
    try:
        conn, cursor = get_db()
        cursor.execute("SELECT id, question, filename, created_at FROM chat_history ORDER BY id DESC LIMIT 20")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return {"history": [
            {"id": r[0], "question": r[1], "filename": r[2], "time": str(r[3])} for r in rows
        ]}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    try:
        conn, cursor = get_db()
        cursor.execute("SELECT COUNT(*) FROM chat_history")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
    except:
        count = 0
    return {
        "service": "PDF 知识库问答系统（带 MySQL）",
        "chat_records": count,
        "status": f"已上传 {len(doc_chunks)} 个知识块" if doc_chunks else "等待上传 PDF"
    }

if __name__ == "__main__":
    print("== PDF 问答系统 + MySQL 启动 ==")
    print("== http://127.0.0.1:18003 ==")
    uvicorn.run(app, host="127.0.0.1", port=18003)
