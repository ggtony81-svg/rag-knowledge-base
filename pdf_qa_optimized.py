"""
PDF 问答 + RAG 优化版（知识库持久化）
"""
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn, os, json, time, requests, numpy as np, pymysql, pickle

import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.remove(script_dir)
import redis as redis_lib
sys.path.insert(0, script_dir)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
from config import DEEPSEEK_API_KEY
from text_chunk import smart_chunk_v2

# 线上用评测里验证过的"分块 + 领域微调"方案：通用模型 P@1 只有 85.9%，
# 换成微调模型并改用 v2 分块后是 92.4%。
#
# 默认加载微调模型，找不到就直接让启动报错，不做"悄悄退回通用模型"的兜底：
# 那样指标掉回去了也不会有人发现。
# 想跳过微调（CPU 上要 3.6 小时）先跑起来看效果，显式指定通用模型即可：
#     MODEL_PATH=BAAI/bge-small-zh-v1.5 python pdf_qa_optimized.py
MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(script_dir, "eval", "cache", "finetuned_model"))
CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS = 400, 50

print(f"加载 embedding 模型: {MODEL_PATH}")
model = SentenceTransformer(MODEL_PATH)
print("模型就绪")

doc_chunks, doc_vectors = [], None
KB_DIR = "d:/shuqi/knowledge_base"
UPLOAD_DIR = "d:/shuqi/uploads"
for d in [KB_DIR, UPLOAD_DIR]: os.makedirs(d, exist_ok=True)
r = redis_lib.Redis(host="localhost", port=6379, decode_responses=True)
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

def kb_signature():
    """当前知识库的"身份"：换模型或换分块参数后，旧向量就不能再用了。"""
    return {
        "model": os.path.abspath(MODEL_PATH),
        "max_tokens": CHUNK_MAX_TOKENS,
        "overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "dim": int(model.get_embedding_dimension()),
    }

def save_kb(fn):
    with open(f"{KB_DIR}/chunks.pkl", "wb") as f: pickle.dump(doc_chunks, f)
    np.save(f"{KB_DIR}/vectors.npy", doc_vectors)
    with open(f"{KB_DIR}/meta.json", "w", encoding="utf-8") as f:
        json.dump({"filename": fn, **kb_signature()}, f, ensure_ascii=False)

def load_kb():
    """
    读取落盘的向量库。

    落盘数据必须带上"由哪个模型、哪套分块参数生成"的指纹：换了模型却
    沿用旧向量，相似度照样算得出来（维度都是 512），只是算出来的是错的
    —— 不报错、不崩溃，静默给出错误排序。
    """
    global doc_chunks, doc_vectors, current_filename
    try:
        with open(f"{KB_DIR}/meta.json", encoding="utf-8") as f:
            info = json.load(f)
        now = kb_signature()
        old = {k: info.get(k) for k in now}
        if old != now:
            print("知识库与当前配置不匹配，已忽略（请重新上传 PDF）")
            print(f"  落盘: {old}")
            print(f"  当前: {now}")
            return
        with open(f"{KB_DIR}/chunks.pkl", "rb") as f: doc_chunks = pickle.load(f)
        doc_vectors = np.load(f"{KB_DIR}/vectors.npy")
        current_filename = info["filename"]
        print(f"加载知识库: {current_filename} ({len(doc_chunks)} 块)")
    except FileNotFoundError:
        print("无持久化知识库，请上传 PDF")
    except Exception as e:
        print(f"知识库加载失败，已忽略: {e}")

def retrieve(question, top_k=3):
    """
    向量检索：把问题编码后和所有块算相似度，返回最相关的 top_k 块。

    块向量和问题向量都做过归一化，所以点积就等于余弦相似度，
    一次矩阵乘法算完 —— 原来是逐块跑 Python 循环，块一多就明显变慢。
    """
    t1 = time.time()
    # 不做 CrossEncoder 重排序：P@1 与"分块+微调"完全相同（都是 92.4%）却慢 365 倍（6.3s vs 17ms）
    qv = model.encode([question], normalize_embeddings=True)[0]
    scores = doc_vectors @ qv
    top = np.argsort(scores)[-top_k:][::-1]
    results = [(doc_chunks[i], round(float(scores[i]), 4)) for i in top]
    return results, round((time.time() - t1) * 1000)

app = FastAPI(title="PDF 问答优化版")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
current_filename = "未命名"
load_kb()

@app.get("/kb/status")
def kb_status():
    """知识库状态。带上模型信息，一眼能看出线上到底在用哪个模型。"""
    return {"loaded": len(doc_chunks) > 0, "filename": current_filename,
            "chunks": len(doc_chunks), "model": os.path.basename(MODEL_PATH),
            "embed_dim": int(model.get_embedding_dimension())}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global doc_chunks, doc_vectors, current_filename
    current_filename = file.filename
    p = os.path.join(UPLOAD_DIR, file.filename)
    with open(p, "wb") as f: f.write(await file.read())
    import fitz
    doc = fitz.open(p); t = "".join(page.get_text() for page in doc)
    n = len(doc); doc.close()
    doc_chunks = smart_chunk_v2(t, max_tokens=CHUNK_MAX_TOKENS,
                               overlap_tokens=CHUNK_OVERLAP_TOKENS,
                               tokenizer=model.tokenizer)
    if not doc_chunks:
        return {"error": "这份 PDF 没有提取到可用文字（可能是扫描件或纯图片），换一份试试"}
    doc_vectors = model.encode(doc_chunks, normalize_embeddings=True)
    save_kb(current_filename)
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
    results, _ = retrieve(req.question)
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

@app.delete("/conversation/{cid}")
def del_conv(cid: int):
    try:
        conn, cursor = get_db()
        cursor.execute("DELETE FROM messages WHERE conversation_id=%s", (cid,))
        cursor.execute("DELETE FROM conversations WHERE id=%s", (cid,))
        conn.commit(); cursor.close(); conn.close()
        return {"status": "ok"}
    except Exception as e: return {"error": str(e)}
@app.post("/chat")
def chat(req: ChatRequest):
    conv_id = req.conversation_id
    history = []
    try:
        conn, cursor = get_db()
        if conv_id is None:
            cursor.execute("INSERT INTO conversations (title, filename) VALUES (%s, %s)", (req.question[:30], current_filename if current_filename else "chat"))
            conn.commit(); conv_id = cursor.lastrowid
        cursor.execute("INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)", (conv_id, "user", req.question))
        conn.commit()
        cursor.execute("SELECT role, content FROM messages WHERE conversation_id=%s ORDER BY id", (conv_id,))
        history = cursor.fetchall()
        cursor.close(); conn.close()
    except: pass
    msgs = [{"role": "system", "content": "你是智能助手。"}]
    for row in history:
        msgs.append({"role": row[0], "content": row[1]})
    def gen():
        full = ""
        resp = requests.post("https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
            json={"model": "deepseek-chat", "messages": msgs, "stream": True, "temperature": 0.7, "max_tokens": 2048}, stream=True, timeout=30)
        for line in resp.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    try:
                        c = json.loads(line[6:])["choices"][0].get("delta",{}).get("content","")
                        if c: full += c; yield c
                    except: pass
        try:
            conn, cursor = get_db()
            cursor.execute("INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)", (conv_id, "assistant", full))
            cursor.execute("UPDATE conversations SET title=(SELECT content FROM messages WHERE conversation_id=%s AND role='user' ORDER BY id LIMIT 1) WHERE id=%s AND title='新对话'", (conv_id, conv_id))
            conn.commit(); cursor.close(); conn.close()
        except: pass
    return StreamingResponse(gen(), media_type="text/plain")

@app.post("/conversation/new")
def new_conv():
    try:
        conn, cursor = get_db()
        cursor.execute("INSERT INTO conversations (title, filename) VALUES (%s, %s)", ("新对话", current_filename))
        conn.commit(); cid = cursor.lastrowid; cursor.close(); conn.close()
        return {"conversation_id": cid}
    except Exception as e: return {"error": str(e)}

WEB_DIR = os.path.join(script_dir, "web")
if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

@app.get("/")
def index():
    """返回问答界面。"""
    return FileResponse(os.path.join(WEB_DIR, "index.html"))

@app.get("/status")
def status():
    return {"status": "ok", "cache_size": r.dbsize(), "kb": len(doc_chunks) > 0,
            "file": current_filename, "model": os.path.basename(MODEL_PATH)}

def sse(obj):
    """把字典包装成一条 SSE 消息。"""
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"

@app.post("/ask/v2")
def ask_v2(req: ChatRequest):
    """
    结构化问答接口，供前端界面调用。

    与 /ask/stream 的区别：除了流式推送回答正文，还会先推一条包含
    "检索到的原文片段"和"各阶段耗时"的消息。这样界面能展示答案依据，
    也能把向量检索、模型生成两段的耗时分别测出来。

    SSE 消息类型：
        meta    会话 id
        sources 检索到的片段 + 向量检索耗时
        delta   回答正文片段
        done    生成耗时 + 端到端耗时
        error   出错信息
    """
    if not doc_chunks:
        return StreamingResponse(iter([sse({"type": "error", "message": "请先上传 PDF 建立知识库"})]),
                                 media_type="text/event-stream")

    conv_id = req.conversation_id
    if conv_id is None:
        try:
            conn, cursor = get_db()
            cursor.execute("INSERT INTO conversations (title, filename) VALUES (%s, %s)",
                           (req.question[:30], current_filename))
            conn.commit(); conv_id = cursor.lastrowid
            cursor.close(); conn.close()
        except Exception:
            conv_id = None

    def save_msg(role, content):
        if conv_id is None: return
        try:
            conn, cursor = get_db()
            cursor.execute("INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)",
                           (conv_id, role, content))
            conn.commit(); cursor.close(); conn.close()
        except Exception:
            pass

    save_msg("user", req.question)

    def gen():
        t0 = time.time()
        yield sse({"type": "meta", "conv_id": conv_id})

        cache_key = f"qa:{req.question}"
        try:
            cached = r.get(cache_key)
        except Exception:
            cached = None

        # 命中缓存：检索都不用做，直接返回上次的回答
        if cached:
            yield sse({"type": "sources", "cached": True, "items": [],
                       "t": {"retrieve_ms": 0}})
            yield sse({"type": "delta", "text": cached})
            save_msg("assistant", cached)
            yield sse({"type": "done", "t": {"total_ms": round((time.time() - t0) * 1000)}})
            return

        # 第一段：向量检索，取相似度最高的 3 块
        results, retrieve_ms = retrieve(req.question)
        yield sse({"type": "sources", "cached": False,
                   "items": [{"text": d, "score": s} for d, s in results],
                   "t": {"retrieve_ms": retrieve_ms}})

        # 第二段：把检索到的资料交给大模型生成回答
        ctx = "\n".join([d for d, _ in results])
        t3 = time.time()
        full_answer = ""
        resp = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": "deepseek-chat",
                  "messages": [{"role": "system", "content": f"基于以下资料回答问题：\n{ctx}"},
                               {"role": "user", "content": req.question}],
                  "stream": True, "temperature": 0.7, "max_tokens": 2048},
            stream=True, timeout=60)
        for line in resp.iter_lines():
            if not line: continue
            line = line.decode("utf-8")
            if line.startswith("data: ") and line[6:] != "[DONE]":
                try:
                    c = json.loads(line[6:])["choices"][0].get("delta", {}).get("content", "")
                except Exception:
                    continue
                if c:
                    full_answer += c
                    yield sse({"type": "delta", "text": c})

        gen_ms = round((time.time() - t3) * 1000)
        try:
            r.setex(cache_key, 3600, full_answer)
        except Exception:
            pass
        save_msg("assistant", full_answer)
        yield sse({"type": "done",
                   "t": {"gen_ms": gen_ms, "total_ms": round((time.time() - t0) * 1000)}})

    return StreamingResponse(gen(), media_type="text/event-stream")

if __name__ == "__main__":
    print("== PDF 问答优化版启动 ==")
    print(f"   embedding 模型: {os.path.basename(MODEL_PATH)}")
    print("   向量检索 + 知识库持久化")
    print("== http://127.0.0.1:18005 ==")
    uvicorn.run(app, host="127.0.0.1", port=18005)
