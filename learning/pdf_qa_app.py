import streamlit as st
import requests, time

API_URL = "http://127.0.0.1:18005"

st.set_page_config(page_title="AI 知识库", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.stApp { background: #fff; }
.block-container { max-width: 100% !important; padding: 1rem 2rem !important; }
.stChatMessage { background: #f5f5f5; border-radius: 14px; padding: 12px 16px; margin: 6px 0; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background: #e8f0fe; }
.stChatInputContainer { border: 1px solid #ddd !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

for k in ["messages", "filename", "conv_id"]:
    if k not in st.session_state:
        st.session_state[k] = None if k == "conv_id" else ([] if k == "messages" else None)

# 侧边栏
with st.sidebar:
    st.markdown("🧠 **AI 知识库**")
    try:
        if requests.get(API_URL, timeout=2).ok: st.markdown("🟢 正常")
    except: st.markdown("🔴 未连接")
    st.markdown("---")
    st.markdown("**文档**")
    up = st.file_uploader("上传 PDF", type=["pdf"], label_visibility="collapsed")
    if up:
        with st.spinner("解析中..."):
            r = requests.post(f"{API_URL}/upload", files={"file": (up.name, up.getvalue(), "application/pdf")})
            if r.ok: d = r.json(); st.session_state.filename = up.name; st.success(f"{up.name} ({d['chunks']} 块)")
    if st.session_state.filename: st.caption(f"📄 {st.session_state.filename}")
    st.markdown("---")
    st.markdown("**对话**")
    if st.button("＋ 新对话", use_container_width=True):
        st.session_state.messages = []; st.session_state.conv_id = None; st.rerun()
    try:
        for conv in requests.get(f"{API_URL}/conversations").json().get("conversations", []):
            label = conv["title"][:18] + "..." if len(conv["title"]) > 18 else conv["title"]
            if st.button(label, key=f"c{conv['id']}", use_container_width=True):
                st.session_state.conv_id = conv["id"]
                r2 = requests.get(f"{API_URL}/conversation/{conv['id']}")
                if r2.ok: st.session_state.messages = r2.json().get("messages", [])
                st.rerun()
    except: pass

# 聊天区
st.markdown("### 💬 问答")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("输入你的问题..."):
    with st.chat_message("user"): st.markdown(prompt)

    if st.session_state.conv_id is None:
        try:
            cr = requests.post(f"{API_URL}/conversation/new")
            if cr.ok: st.session_state.conv_id = cr.json().get("conversation_id")
        except: pass

    with st.chat_message("assistant"):
        box = st.empty(); answer = ""; start = time.time()
        try:
            resp = requests.post(f"{API_URL}/ask/stream",
                json={"question": prompt, "conversation_id": st.session_state.conv_id},
                stream=True, timeout=30)
            for chunk in resp.iter_content(None):
                if chunk: answer += chunk.decode("utf-8"); box.markdown(answer + "▌")
            tag = "⚡ 缓存" if time.time() - start < 1 else "⏳ 模型生成"
            box.markdown(f"{answer}\n\n`{tag} · {time.time()-start:.1f}s`")
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"请求失败，请检查 API 服务")
