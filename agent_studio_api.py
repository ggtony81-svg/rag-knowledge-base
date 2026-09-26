"""
AI 协作工坊 — 多智能体协作系统
策划师 → 执行者 → 审查员 → 优化者 → 循环迭代
"""
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn, os, json, requests, pymysql, hashlib

import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.remove(script_dir)
import redis as redis_lib
sys.path.insert(0, script_dir)

from config import DEEPSEEK_API_KEY

r = redis_lib.Redis(host="localhost", port=6379, decode_responses=True)

# ============================================================
# Redis 的两个用法
#   1. 分布式锁：防重复提交（一次任务要跑几十秒、烧 token，点两次就跑两份）
#   2. Cache-Aside 缓存：任务详情读多写少，且 content 字段很大
# 所有 Redis 调用都做降级处理：Redis 挂了只损失锁和缓存，主流程照跑
# ============================================================
def acquire_lock(requirement, ttl=300):
    """返回 (是否拿到锁, 锁key)。SET NX EX 是原子的：key 不存在才写入，且自带过期时间——
    进程崩了锁也会自动释放，不会像行锁那样永久占住（死锁兜底）。"""
    key = f"studio:lock:{hashlib.md5(requirement.encode()).hexdigest()}"
    try:
        return (True, key) if r.set(key, "1", nx=True, ex=ttl) else (False, key)
    except Exception as e:
        print(f"[Redis] 不可用，跳过加锁: {e}")
        return (True, None)          # 降级放行，key=None 表示无需释放

def release_lock(key):
    if not key: return
    try: r.delete(key)
    except Exception: pass

def cache_get(key):
    try:
        v = r.get(key)
        return json.loads(v) if v else None
    except Exception: return None

def cache_set(key, value, ttl=300):
    try: r.setex(key, ttl, json.dumps(value, ensure_ascii=False))
    except Exception: pass

def cache_del(key):
    try: r.delete(key)
    except Exception: pass

def get_db():
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='pdf_qa', charset='utf8mb4')
    return conn, conn.cursor()

try:
    conn, cursor = get_db()
    cursor.execute("""CREATE TABLE IF NOT EXISTS studio_tasks (
        id INT AUTO_INCREMENT PRIMARY KEY,
        title VARCHAR(255), task_type VARCHAR(50), requirement TEXT,
        status VARCHAR(20) DEFAULT 'running',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS studio_steps (
        id INT AUTO_INCREMENT PRIMARY KEY,
        task_id INT NOT NULL, agent VARCHAR(50), action VARCHAR(50),
        content TEXT, score INT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (task_id) REFERENCES studio_tasks(id) ON DELETE CASCADE)""")
    conn.commit(); cursor.close(); conn.close()
    print("MySQL 就绪")
except Exception as e: print(f"MySQL: {e}")

app = FastAPI(title="AI 协作工坊")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def call_llm(messages):
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "deepseek-chat", "messages": messages, "temperature": 0.7, "max_tokens": 2048}
    resp = requests.post(url, headers=headers, json=data, timeout=60)
    return resp.json()["choices"][0]["message"]["content"]

# ============================================================
# Agent 角色定义
# ============================================================
PLANNER_PROMPT = """你是一个策划师。分析用户需求，输出一份清晰的执行方案。
格式：
## 目标理解
(用2-3句话说明你要做什么)

## 执行方案
(列出具体步骤和要点)

## 注意事项
(需要特别关注的点)"""

WRITER_PROMPT = """你是一个执行者。根据方案完成任务。
要求：
- 直接输出成果，不要解释过程
- 内容要完整、专业、可直接使用
- 如果是代码，给出完整可运行的代码"""

CRITIC_PROMPT = """你是一个严格的审查员。审查下面的成果，找出问题。
要求：
1. 按严重程度打分（1-10分，10分完美）
2. 如果提供了上一版，请明确对比"相比上一版，这一版改进了什么"
3. 列出具体问题（至少3条，要有针对性，不能是空话）
4. 指出可以改进的具体地方

输出格式：
## 评分：X/10
## 相比上一版的改进
## 仍然存在的问题
## 改进建议"""

OPTIMIZER_PROMPT = """你是一个优化者。根据审查意见修改成果。
要求：
- 修复审查员指出的问题
- 保留原成果的优点
- 输出修改后的完整版本，不要解释过程"""

# ============================================================
# 工作流引擎
# ============================================================
def run_workflow(task_id, requirement, task_type, max_rounds=3):
    """多智能体协作循环：策划→执行→审查→优化→再审查"""
    steps = []
    current_work = None
    final_score = 0

    # 1. 策划
    plan = call_llm([
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content": f"任务类型：{task_type}\n需求：{requirement}"}
    ])
    steps.append({"agent": "策划师", "action": "制定方案", "content": plan})
    yield {"event": "step", "agent": "策划师", "action": "制定方案", "content": plan}
    save_step(task_id, "策划师", "制定方案", plan, None)

    # 2. 执行 + 审查 + 优化 循环
    for round_num in range(max_rounds):
        # 本轮开始前的成果 = "上一版"（第0轮是None），交给审查员对比
        prev_work = current_work
        # 执行
        if current_work is None:
            writer_input = f"任务类型：{task_type}\n需求：{requirement}\n\n方案：\n{plan}"
        else:
            writer_input = f"任务类型：{task_type}\n需求：{requirement}\n\n上一版成果：\n{current_work}"
        work = call_llm([
            {"role": "system", "content": WRITER_PROMPT},
            {"role": "user", "content": writer_input}
        ])
        current_work = work
        steps.append({"agent": "执行者", "action": f"第{round_num+1}轮产出", "content": work})
        yield {"event": "step", "agent": "执行者", "action": f"第{round_num+1}轮产出", "content": work}
        save_step(task_id, "执行者", f"第{round_num+1}轮产出", work, None)

        critic_input = f"任务需求：{requirement}\n\n成果：\n{work}"

        if prev_work and round_num > 0:
            critic_input += f"\n\n上一版成果：\n{prev_work}"

        critic = call_llm([
            {"role": "system", "content": CRITIC_PROMPT},
            {"role": "user", "content": critic_input}
        ])
        # 尝试提取分数
        try:
            import re
            m = re.search(r'评分[:：]\s*(\d+)\s*/', critic)
            final_score = int(m.group(1)) if m else 7
        except:
            final_score = 7

        steps.append({"agent": "审查员", "action": f"第{round_num+1}轮审查", "content": critic})
        yield {"event": "step", "agent": "审查员", "action": f"第{round_num+1}轮审查", "content": critic}
        save_step(task_id, "审查员", f"第{round_num+1}轮审查", critic, final_score)

        # 如果分数够高，停止迭代
        # 强制至少2轮，分数够高才提前结束
        if final_score >= 8 and round_num >= 1:
            yield {"event": "done", "score": final_score, "rounds": round_num+1}
            return

        # 优化
        optimized = call_llm([
            {"role": "system", "content": OPTIMIZER_PROMPT},
            {"role": "user", "content": f"任务需求：{requirement}\n\n原成果：\n{work}\n\n审查意见：\n{critic}"}
        ])
        current_work = optimized
        steps.append({"agent": "优化者", "action": f"第{round_num+1}轮优化", "content": optimized})
        yield {"event": "step", "agent": "优化者", "action": f"第{round_num+1}轮优化", "content": optimized}
        save_step(task_id, "优化者", f"第{round_num+1}轮优化", optimized, None)

        # 最后一轮后不再循环
        if round_num == max_rounds - 1:
            yield {"event": "done", "score": final_score, "rounds": round_num+1}

def save_step(task_id, agent, action, content, score):
    try:
        conn, cursor = get_db()
        cursor.execute("INSERT INTO studio_steps (task_id, agent, action, content, score) VALUES (%s, %s, %s, %s, %s)",
            (task_id, agent, action, content, score))
        conn.commit(); cursor.close(); conn.close()
        cache_del(f"studio:task:{task_id}")   # 写后失效：下次读会从 MySQL 重建，保证不读到旧数据
    except: pass

# ============================================================
# API
# ============================================================
class TaskRequest(BaseModel):
    task_type: str
    requirement: str

@app.post("/studio/run")
def run_studio(req: TaskRequest):
    """启动多智能体协作"""
    # 防重复提交：同一个需求正在跑时，直接挡掉
    ok, lock_key = acquire_lock(req.requirement)
    if not ok:
        return {"error": "该需求正在执行中，请勿重复提交"}

    try:
        conn, cursor = get_db()
        cursor.execute("INSERT INTO studio_tasks (title, task_type, requirement) VALUES (%s, %s, %s)",
            (req.requirement[:30], req.task_type, req.requirement))
        conn.commit()
        task_id = cursor.lastrowid
        cursor.close(); conn.close()
    except:
        task_id = 1

    def gen():
        try:
            for event in run_workflow(task_id, req.requirement, req.task_type):
                yield json.dumps(event, ensure_ascii=False) + "\n"
            # 更新状态
            try:
                conn, cursor = get_db()
                cursor.execute("UPDATE studio_tasks SET status='done' WHERE id=%s", (task_id,))
                conn.commit(); cursor.close(); conn.close()
            except: pass
            cache_del(f"studio:task:{task_id}")
        finally:
            release_lock(lock_key)   # 跑完/中途断流/报错，都要放锁

    return StreamingResponse(gen(), media_type="application/json")

@app.get("/studio/tasks")
def list_tasks():
    try:
        conn, cursor = get_db()
        cursor.execute("SELECT id, title, task_type, status, created_at FROM studio_tasks ORDER BY id DESC")
        rows = cursor.fetchall(); cursor.close(); conn.close()
        return {"tasks": [{"id": r[0], "title": r[1], "type": r[2], "status": r[3], "time": str(r[4])} for r in rows]}
    except: return {"tasks": []}

@app.get("/studio/task/{tid}")
def get_task(tid: int):
    cache_key = f"studio:task:{tid}"
    # Cache-Aside：先查缓存，命中直接返回
    hit = cache_get(cache_key)
    if hit is not None:
        return {**hit, "from_cache": True}

    try:
        conn, cursor = get_db()
        cursor.execute("SELECT agent, action, content, score, created_at FROM studio_steps WHERE task_id=%s ORDER BY id", (tid,))
        rows = cursor.fetchall(); cursor.close(); conn.close()
        data = {"steps": [{"agent": r[0], "action": r[1], "content": r[2], "score": r[3], "time": str(r[4])} for r in rows]}
        cache_set(cache_key, data)      # 回填缓存，TTL 300 秒
        return {**data, "from_cache": False}
    except: return {"steps": [], "from_cache": False}

@app.delete("/studio/task/{tid}")
def del_task(tid: int):
    try:
        conn, cursor = get_db()
        cursor.execute("DELETE FROM studio_steps WHERE task_id=%s", (tid,))
        cursor.execute("DELETE FROM studio_tasks WHERE id=%s", (tid,))
        conn.commit(); cursor.close(); conn.close()
        cache_del(f"studio:task:{tid}")   # 删了任务必须清缓存，否则会返回已删除的数据
        return {"status": "ok"}
    except: return {"error": "failed"}

@app.get("/")
def root():
    try: keys = r.dbsize()
    except Exception: keys = -1
    return {"service": "AI 协作工坊", "status": "running", "redis_keys": keys}

if __name__ == "__main__":
    print("== AI 协作工坊启动 ==")
    print("== http://127.0.0.1:18007 ==")
    uvicorn.run(app, host="127.0.0.1", port=18007)
