"""
流式输出演示 — 像 ChatGPT 一样逐字显示
"""
import requests
import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from config import DEEPSEEK_API_KEY

def stream_chat(messages):
    """流式调 DeepSeek，一个字一个字地 yield"""
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "stream": True,              # ← 关键：开启流式
        "temperature": 0.7,
        "max_tokens": 1024
    }

    response = requests.post(url, headers=headers, json=data, stream=True, timeout=30)

    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data_str = line[6:]  # 去掉 "data: " 前缀
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
# 对比演示
# ============================================================
question = "用30个字以内介绍一下什么是RAG"

print("=" * 60)
print("🙋 问：什么是RAG？")
print("=" * 60)
print()
print("🤖 流式输出效果（逐字显示）:")
print("  ", end="", flush=True)

# 流式输出
full_reply = ""
for chunk in stream_chat([
    {"role": "user", "content": question}
]):
    print(chunk, end="", flush=True)
    full_reply += chunk

print()
print()
print("=" * 60)
print("💡 流式 vs 普通输出的区别")
print()
print("  普通输出：请求 → 等3-5秒 → 一次性看到全部文字")
print("  流式输出：请求 → 0.5秒后开始逐字显示 ← 用户体验更好")
print()
print("  原理：模型生成一个字，服务器就推送一个字")
print("        浏览器/客户端收到一个字就显示一个字")
print("        不需要等全部生成完")
