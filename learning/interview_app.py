"""
AI 面试官前端
"""
import streamlit as st
import requests, time

API_URL = "http://127.0.0.1:18006"

st.set_page_config(page_title="AI 面试官", page_icon="🎤", layout="wide")

st.markdown("""
<style>
.stApp { background: #f8f9fa; }
.block-container { max-width: 100% !important; padding: 1rem 2rem !important; }
.stChatMessage { background: #fff; border-radius: 12px; padding: 12px 16px; margin: 6px 0; border: 1px solid #e8e8e8; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background: #e8f0fe; border-color: #d2e3fc; }
</style>
""", unsafe_allow_html=True)

for k in ["messages", "interview_id", "page", "setup_done"]:
    if k not in st.session_state:
        if k == "page": st.session_state[k] = "start"
        elif k == "messages": st.session_state[k] = []
        elif k == "setup_done": st.session_state[k] = False
        else: st.session_state[k] = None

with st.sidebar:
    st.markdown("🎤 **AI 面试官**")
    pages = {"start": "🎯 开始面试", "interview": "💬 面试中", "history": "📋 历史"}
    for k, v in pages.items():
        t = "primary" if st.session_state.page == k else "secondary"
        if st.button(v, use_container_width=True, type=t): st.session_state.page = k; st.rerun()

page = st.session_state.page

# ============================================================
# 开始面试
# ============================================================
if page == "start":
    st.markdown("## 🎯 开始一场 AI 面试")
    st.markdown("粘贴目标岗位 JD 和你的简历，AI 面试官会生成面试计划并开始提问")

    c1, c2 = st.columns(2)
    with c1:
        jd = st.text_area("岗位 JD", height=280, placeholder="粘贴目标岗位描述...")
    with c2:
        resume = st.text_area("你的简历", height=280, placeholder="粘贴简历内容...")

    if st.button("🎤 开始面试", type="primary", use_container_width=True):
        if jd.strip() and resume.strip():
            with st.spinner("AI 面试官正在分析岗位和简历..."):
                try:
                    resp = requests.post(f"{API_URL}/interview/setup",
                        json={"job_desc": jd, "resume": resume}, timeout=60)
                    if resp.ok:
                        data = resp.json()
                        st.session_state.interview_id = data["interview_id"]
                        st.session_state.messages = [{"role": "assistant", "content": data["setup"]}]
                        st.session_state.setup_done = True
                        st.session_state.page = "interview"
                        st.rerun()
                    else:
                        st.error("启动失败，请确保 API 已启动 (python interview_api.py)")
                except Exception as e:
                    st.error(f"连接失败: {e}")
        else:
            st.warning("请填写 JD 和简历")

# ============================================================
# 面试中
# ============================================================
elif page == "interview":
    st.markdown("## 💬 面试进行中")
    if st.session_state.interview_id is None:
        st.warning("还没有开始面试，请先到「开始面试」页面")
    else:
        # 显示对话
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        # 结束面试按钮
        col1, col2 = st.columns([5,1])
        with col2:
            if st.button("🏁 结束面试", use_container_width=True):
                st.session_state.messages = []
                st.session_state.interview_id = None
                st.session_state.page = "start"
                st.rerun()

        # 输入回答
        if ans := st.chat_input("回答面试官的问题..."):
            with st.chat_message("user"): st.markdown(ans)
            st.session_state.messages.append({"role": "user", "content": ans})

            with st.chat_message("assistant"):
                box = st.empty(); full = ""
                try:
                    resp = requests.post(f"{API_URL}/interview/ask",
                        json={"interview_id": st.session_state.interview_id, "answer": ans},
                        stream=True, timeout=60)
                    for chunk in resp.iter_content(None):
                        if chunk:
                            full += chunk.decode("utf-8")
                            box.markdown(full + "▌")
                    box.markdown(full)
                    st.session_state.messages.append({"role": "assistant", "content": full})
                    st.rerun()
                except:
                    st.error("连接失败，请确保 API 运行中")

# ============================================================
# 历史
# ============================================================
elif page == "history":
    st.markdown("## 📋 面试历史")
    try:
        resp = requests.get(f"{API_URL}/interview/history", timeout=3)
        interviews = resp.json().get("interviews", [])
        if not interviews:
            st.info("还没有面试记录")
        for iv in interviews:
            cols = st.columns([4, 1])
            with cols[0]:
                if st.button(f"#{iv['id']} {iv['title']} — {iv['time']}", key=f"iv{iv['id']}", use_container_width=True):
                    r2 = requests.get(f"{API_URL}/interview/{iv['id']}/messages", timeout=3)
                    if r2.ok:
                        st.session_state.messages = r2.json().get("messages", [])
                        st.session_state.interview_id = iv["id"]
                        st.session_state.page = "interview"
                        st.rerun()
            with cols[1]:
                if st.button("删除", key=f"del{iv['id']}"):
                    requests.delete(f"{API_URL}/interview/{iv['id']}")
                    st.rerun()
    except:
        st.error("服务未连接")
