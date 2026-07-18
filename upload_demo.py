"""
FastAPI 文件上传演示
接收用户上传的 PDF，保存到本地
"""
from fastapi import FastAPI, UploadFile, File
import uvicorn
import os

app = FastAPI(title="文件上传演示")

UPLOAD_DIR = "d:/shuqi/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传 PDF 文件"""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # 用 PyMuPDF 读取内容
    import fitz
    doc = fitz.open(file_path)
    num_pages = len(doc)         # ← 先记下页数
    total_chars = sum(len(page.get_text()) for page in doc)
    doc.close()                  # ← 再关闭

    return {
        "filename": file.filename,
        "size": len(content),
        "pages": num_pages,
        "total_chars": total_chars,
        "saved_to": file_path
    }

@app.get("/")
def root():
    return {
        "usage": "POST /upload 上传 PDF",
        "files": os.listdir(UPLOAD_DIR) if os.path.exists(UPLOAD_DIR) else []
    }

if __name__ == "__main__":
    print("== 文件上传服务启动 ==")
    print("== http://127.0.0.1:18002/docs ==")
    uvicorn.run(app, host="127.0.0.1", port=18002)
