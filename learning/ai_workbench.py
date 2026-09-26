"""
AI 工作台
"""
import streamlit as st
import requests, json, time

API_URL = "http://127.0.0.1:18005"

st.set_page_config(page_title="AI 工作台", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    /* 全局 */
    .stApp { background: #f0f2f5; }
    .block-container { max-width: 100% !important; padding: 0 !important; }

    /* 顶部导航 */
    .nav { background: #fff; border-bottom: 1px solid #e8e8e8; padding: 0 24px; display: flex; align-items: center; height: 52px; position: fixed; top: 0; left: 0; right: 0; z-index: 100; }
    .nav-title { font-size: 16px; font-weight: 600; color: #1a1a1a; margin-right: 32px; }
    .nav-btn { background: none; border: none; padding: 0 16px; height: 52px; cursor: pointer; font-size: 14px; color: #666; border-bottom: 2px solid transparent; }
    .nav-btn:hover { color: #1a1a1a; background: #f5f5f5; }
    .nav-btn.active { color: #1a73e8; border-bottom-color: #1a73e8; font-weight: 500; }

    /* 主布局 */
    .main { display: flex; height: calc(100vh - 52px); margin-top: 52px; }

    /* 侧栏 */
    .sidebar { width: 260px; background: #fff; border-right: 1px solid #e8e8e8; padding: 12px; overflow-y: auto; flex-shrink: 0; }
    .sidebar-title { font-size: 12px; color: #999; text-transform: uppercase; letter-spacing: 0.5px; padding: 8px 12px 4px; }
    .conv-item { display: flex; align-items: center; padding: 8px 12px; border-radius: 8px; cursor: pointer; font-size: 13px; color: #333; margin: 1px 0; }
    .conv-item:hover { background: #f0f2f5; }
    .conv-item.active { background: #e8f0fe; color: #1a73e8; }
    .conv-title { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .conv-del { background: none; border: none; color: #ccc; cursor: pointer; padding: 2px 4px; border-radius: 4px; font-size: 12px; }
    .conv-del:hover { color: #e94560; background: #fce8e6; }

    /* 聊天区 */
    .chat-area { flex: 1; display: flex; flex-direction: column; background: #f0f2f5; }
    .chat-header { padding: 16px 24px; background: #fff; border-bottom: 1px solid #e8e8e8; font-size: 15px; font-weight: 500; color: #1a1a1a; }
    .chat-messages { flex: 1; overflow-y: auto; padding: 16px 24px; }
    .chat-input-area { padding: 12px 24px 16px; background: #fff; border-top: 1px solid #e8e8e8; }

    /* 气泡 */
    .bubble { max-width: 75%; padding: 12px 16px; border-radius: 16px; margin: 6px 0; line-height: 1.5; font-size: 14px; }
    .bubble-user { background: #1a73e8; color: #fff; margin-left: auto; border-bottom-right-radius: 4px; }
    .bubble-assistant { background: #fff; color: #1a1a1a; border: 1px solid #e8e8e8; border-bottom-left-radius: 4px; }
    .bubble-time { font-size: 11px; color: #999; margin-top: 2px; }

    /* 隐藏 Streamlit 默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    div[data-testid="stToolbar"] {display:none;}
    div[data-testid="stDecoration"] {display:none;}

    /* 覆盖 Streamlit 默认 */
    .stButton button { border-radius: 8px; }
    .stTextInput input { border-radius: 8px; border: 1px solid #ddd; }
    div.stChatInputContainer { border: none !important; background: transparent !important; padding: 0 !important; }
    div.stChatFloatingInputContainer { bottom: 0; padding: 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 初始化
# ============================================================
for k in ["messages", "conv_id", "page"]:
    if k not in st.session_state:
        st.session_state[k] = "chat" if k == "page" else ([] if k == "messages" else None)

# ============================================================
# 获取对话列表
# ============================================================
def get_convs():
    try:
        return requests.get(f"{API_URL}/conversations", timeout=3).json().get("conversations", [])
    except: return []

# ============================================================
# 页面
# ============================================================
pages = {"chat": "💬 对话", "rag": "📚 知识库", "resume": "📝 简历"}
page = st.session_state.page

# 侧栏
with st.sidebar:
    st.markdown("### ⚡ AI 工作台")
    for k, v in pages.items():
        if st.button(v, use_container_width=True, type="primary" if page == k else "secondary"):
            st.session_state.page = k; st.rerun()
    st.markdown("---")
    st.markdown("**📋 历史对话**")
    if st.button("＋ 新对话", use_container_width=True):
        st.session_state.messages = []; st.session_state.conv_id = None; st.rerun()
    for c in get_convs():
        cols = st.columns([5,1])
        label = c["title"][:20]+"..." if len(c["title"])>20 else c["title"]
        with cols[0]:
            if st.button(label, key=f"s{c['id']}", use_container_width=True):
                st.session_state.conv_id = c["id"]
                r2 = requests.get(f"{API_URL}/conversation/{c['id']}", timeout=3)
                if r2.ok: st.session_state.messages = r2.json().get("messages", [])
                st.rerun()
        with cols[1]:
            if st.button("✕", key=f"d{c['id']}", help="删除"):
                try: requests.delete(f"{API_URL}/conversation/{c['id']}")
                except: pass
                if st.session_state.conv_id == c["id"]:
                    st.session_state.messages = []; st.session_state.conv_id = None
                st.rerun()

# ============================================================
# 对话页面
# ============================================================
if page == "chat":
    st.markdown("### 💬 对话")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if p := st.chat_input("输入消息..."):
        with st.chat_message("user"): st.markdown(p)
        with st.chat_message("assistant"):
            box = st.empty(); a = ""; start = time.time()
            try:
                if st.session_state.conv_id is None:
                    cr = requests.post(f"{API_URL}/conversation/new", timeout=5)
                    if cr.ok: st.session_state.conv_id = cr.json()["conversation_id"]
                resp = requests.post(f"{API_URL}/chat", json={"question":p, "conversation_id":st.session_state.conv_id}, stream=True, timeout=30)
                for c in resp.iter_content(None):
                    if c: a += c.decode("utf-8"); box.markdown(a + "▌")
                box.markdown(f"{a}\n\n`{time.time()-start:.1f}s`")
                st.session_state.messages.append({"role":"user","content":p})
                st.session_state.messages.append({"role":"assistant","content":a})
                st.rerun()
            except: st.error("连接失败")

# ============================================================
# 知识库页面
# ============================================================
elif page == "rag":
    st.markdown("### 📚 知识库")
    try:
        ks = requests.get(f"{API_URL}/kb/status", timeout=3).json()
        if ks.get("loaded"): st.info(f"📄 {ks['filename']} ({ks['chunks']} 块)")
        else: st.warning("知识库为空")
    except: st.warning("服务未启动")
    up = st.file_uploader("上传 PDF", type=["pdf"], key="rag_up")
    if up:
        with st.spinner("解析中..."):
            r = requests.post(f"{API_URL}/upload", files={"file":(up.name,up.getvalue(),"application/pdf")}, timeout=30)
            if r.ok: st.success(f"{up.name} 已加载"); st.rerun()
            else: st.error("上传失败")
    for m in st.session_state.messages:
        if m["role"] in ["user","assistant"]:
            with st.chat_message(m["role"]): st.markdown(m["content"])
    if q := st.chat_input("基于文档提问..."):
        with st.chat_message("user"): st.markdown(q)
        with st.chat_message("assistant"):
            box = st.empty(); a = ""; start = time.time()
            try:
                resp = requests.post(f"{API_URL}/ask/stream", json={"question":q}, stream=True, timeout=30)
                for c in resp.iter_content(None):
                    if c: a += c.decode("utf-8"); box.markdown(a + "▌")
                box.markdown(f"{a}\n\n`{time.time()-start:.1f}s`")
            except: st.error("请上传 PDF")

# ============================================================
# 简历页面
# ============================================================
elif page == "resume":
    st.markdown("### 📝 简历优化")
    c1, c2 = st.columns(2)
    with c1:
        r = st.text_area("粘贴简历", height=300, placeholder="教育、经历、技能...")
        j = st.text_area("目标岗位（可选）", height=150, placeholder="JD...")
        if st.button("🚀 优化", type="primary", use_container_width=True) and r.strip():
            with st.spinner("分析中..."):
                from config import DEEPSEEK_API_KEY
                t = f"目标岗位：{j}" if j else ""
                p = f"你是一针见血的简历优化专家。{t}原文：{r}要求：删套话、改空洞、量化成果。输出格式：## 问题\n## 优化版\n## 最重要的3处改动"
                resp = requests.post("https://api.deepseek.com/chat/completions",
                    headers={"Authorization":f"Bearer {DEEPSEEK_API_KEY}"},
                    json={"model":"deepseek-chat","messages":[{"role":"system","content":"你是毒舌简历优化专家。"},{"role":"user","content":p}],"temperature":0.7,"max_tokens":2048}, timeout=30)
                st.session_state.resume_result = resp.json()["choices"][0]["message"]["content"]
    with c2:
        if "resume_result" in st.session_state: st.markdown(st.session_state.resume_result)
