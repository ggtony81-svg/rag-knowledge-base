"""
AI 协作工坊前端
"""
import streamlit as st
import requests, json

API_URL = "http://127.0.0.1:18007"

st.set_page_config(page_title="AI 协作工坊", page_icon="🏗️", layout="wide")

st.markdown("""
<style>
.stApp { background: #f8f9fa; }
.block-container { max-width: 100% !important; padding: 1rem 2rem !important; }
.agent-card { background: #fff; border-radius: 12px; padding: 16px; border: 1px solid #e8e8e8; margin: 8px 0; }
.agent-badge { display: inline-block; padding: 3px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; margin-bottom: 8px; }
.agent-planner { background: #e8f0fe; color: #1a73e8; }
.agent-writer { background: #e6f4ea; color: #188038; }
.agent-critic { background: #fce8e6; color: #d93025; }
.agent-optimizer { background: #fef7e0; color: #b06000; }
.score-badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 13px; font-weight: 700; margin-left: 8px; }
</style>
""", unsafe_allow_html=True)

for k in ["page"]:
    if k not in st.session_state: st.session_state[k] = "run"

with st.sidebar:
    st.markdown("🏗️ **AI 协作工坊**")
    pages = {"run": "🚀 开始任务", "history": "📋 历史记录"}
    for k, v in pages.items():
        t = "primary" if st.session_state.page == k else "secondary"
        if st.button(v, use_container_width=True, type=t): st.session_state.page = k; st.rerun()

page = st.session_state.page

# ============================================================
# 开始任务
# ============================================================
if page == "run":
    st.markdown("## 🏗️ AI 协作工坊")
    st.markdown("**策划师 → 执行者 → 审查员 → 优化者 → 循环迭代**，一个任务由多个 AI 角色协作完成")

    c1, c2 = st.columns([1, 2])
    with c1:
        task_type = st.selectbox("任务类型", ["写代码", "写文章", "写方案", "写文案"])
    with c2:
        requirement = st.text_area("任务需求", height=100, placeholder="例：用 Python 写一个批量重命名文件的脚本")

    if st.button("🚀 启动协作", type="primary", use_container_width=True):
        if requirement.strip():
            st.session_state.running = True
            st.session_state.task_type = task_type
            st.session_state.requirement = requirement
            st.rerun()

    if st.session_state.get("running"):
        st.markdown("---")
        st.markdown("### 协作过程")

        final_score = None
        rounds = 0

        try:
            resp = requests.post(f"{API_URL}/studio/run",
                json={"task_type": st.session_state.task_type, "requirement": st.session_state.requirement},
                stream=True, timeout=300)

            for line in resp.iter_lines():
                if line:
                    try:
                        event = json.loads(line.decode("utf-8"))
                    except:
                        continue

                    if event["event"] == "step":
                        agent = event["agent"]
                        badge_class = {
                            "策划师": "agent-planner", "执行者": "agent-writer",
                            "审查员": "agent-critic", "优化者": "agent-optimizer"
                        }.get(agent, "agent-planner")
                        icons = {"策划师": "📋", "执行者": "🔨", "审查员": "🔍", "优化者": "🔧"}

                        with st.expander(f"{icons[agent]} {agent} — {event['action']}", expanded=True):
                            st.markdown(f'<span class="agent-badge {badge_class}">{agent}</span>', unsafe_allow_html=True)
                            st.markdown(event["content"])

                    elif event["event"] == "done":
                        final_score = event["score"]
                        rounds = event["rounds"]

            if final_score is not None:
                color = "#188038" if final_score >= 8 else ("#b06000" if final_score >= 6 else "#d93025")
                st.markdown(f"""
                <div style="background:#fff;border-radius:12px;padding:20px;border:1px solid #e8e8e8;margin-top:16px;text-align:center">
                    <div style="font-size:14px;color:#666">协作完成 · 经过 {rounds} 轮迭代</div>
                    <div style="font-size:48px;font-weight:700;color:{color};margin:8px 0">{final_score}/10</div>
                    <div style="font-size:13px;color:#999">最终评分</div>
                </div>
                """, unsafe_allow_html=True)

            st.session_state.running = False
        except Exception as e:
            st.error(f"连接失败: {e}")
            st.info("请确保 API 已启动: python agent_studio_api.py")

# ============================================================
# 历史
# ============================================================
elif page == "history":
    st.markdown("## 📋 历史任务")
    try:
        tasks = requests.get(f"{API_URL}/studio/tasks", timeout=3).json().get("tasks", [])
        if not tasks:
            st.info("还没有任务记录")
        for t in tasks:
            cols = st.columns([4, 1])
            with cols[0]:
                if st.button(f"#{t['id']} {t['title'][:25]} — {t['type']} — {t['time']}", key=f"t{t['id']}", use_container_width=True):
                    r2 = requests.get(f"{API_URL}/studio/task/{t['id']}", timeout=3)
                    if r2.ok:
                        st.session_state.selected_task = r2.json().get("steps", [])
                        st.rerun()
            with cols[1]:
                if st.button("删除", key=f"td{t['id']}"):
                    requests.delete(f"{API_URL}/studio/task/{t['id']}")
                    st.rerun()
    except:
        st.error("服务未连接")

    if "selected_task" in st.session_state and st.session_state.selected_task:
        st.markdown("---")
        st.markdown("### 协作过程详情")
        icons = {"策划师": "📋", "执行者": "🔨", "审查员": "🔍", "优化者": "🔧"}
        colors = {"策划师": "#1a73e8", "执行者": "#188038", "审查员": "#d93025", "优化者": "#b06000"}
        for step in st.session_state.selected_task:
            agent = step["agent"]
            score = step.get("score")
            score_html = f'<span class="score-badge" style="background:#fce8e6;color:#d93025">评分 {score}/10</span>' if score else ""
            st.markdown(f"""
            <div class="agent-card">
                <span style="color:{colors[agent]};font-weight:600">{icons[agent]} {agent} · {step['action']}</span>
                {score_html}
                <div style="margin-top:8px;color:#333;font-size:14px">{step['content']}</div>
                <div style="margin-top:6px;color:#999;font-size:12px">{step['time']}</div>
            </div>
            """, unsafe_allow_html=True)
