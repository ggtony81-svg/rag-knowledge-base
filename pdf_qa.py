"""
PDF 知识库问答系统
上传 PDF → 解析 → 分块 → 向量化 → 检索 → 流式回答
"""
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn, os, json, requests, numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
from config import DEEPSEEK_API_KEY

print("加载模型中...")
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# 知识库（动态构建）
doc_chunks = []       # 存储文本块
doc_vectors = None    # 存储向量

UPLOAD_DIR = "d:/shuqi/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============================================================
# 工具：分块
# ============================================================
def chunk_text(text, chunk_size=200):
    """把长文本切成小块，每块最多 chunk_size 个字"""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    # 按句号、换行、逗号切
    for separator in ["\n\n", "\n", "。", "；", "，"]:
        if separator in text:
            parts = text.split(separator)
            current = ""
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if len(current) + len(part) < chunk_size:
                    current += part + separator
                else:
                    if current:
                        chunks.append(current.strip())
                    current = part + separator
            if current:
                chunks.append(current.strip())
            break
    else:
        # 没找到分隔符，硬切
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])

    return [c for c in chunks if len(c) > 10]

# ============================================================
# API
# ============================================================
app = FastAPI(title="PDF 知识库问答系统")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """上传 PDF，自动构建知识库"""
    global doc_chunks, doc_vectors

    # 保存文件
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # 解析 PDF
    import fitz
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    num_pages = len(doc)
    doc.close()

    # 分块
    doc_chunks = chunk_text(full_text)

    # 向量化
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
    """提问：检索知识库 + 流式回答"""
    if not doc_chunks:
        return StreamingResponse(iter(["请先上传 PDF 文档"]), media_type="text/plain")

    # 向量检索
    q_vec = model.encode([req.question])
    scores = []
    for d_vec in doc_vectors:
        cos_sim = np.dot(q_vec[0], d_vec) / (np.linalg.norm(q_vec[0]) * np.linalg.norm(d_vec))
        scores.append(cos_sim)
    top_idx = np.argsort(scores)[-3:][::-1]
    context = "\n".join([doc_chunks[i] for i in top_idx])

    # 调 DeepSeek 流式回答
    def generate():
        resp = requests.post("https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": f"基于以下资料回答问题，不知道就说不知道：\n{context}"},
                    {"role": "user", "content": req.question}
                ],
                "stream": True,
                "temperature": 0.7,
                "max_tokens": 2048
            }, stream=True, timeout=30)

        for line in resp.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        content = chunk["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                    except:
                        pass

    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/")
def root():
    return {
        "service": "PDF 知识库问答系统",
        "status": f"已上传 {len(doc_chunks)} 个知识块" if doc_chunks else "等待上传 PDF"
    }

if __name__ == "__main__":
    print("== PDF 知识库问答系统启动 ==")
    print("== http://127.0.0.1:18003/docs ==")
    uvicorn.run(app, host="127.0.0.1", port=18003)
