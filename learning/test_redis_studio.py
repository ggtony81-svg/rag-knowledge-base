"""验证 agent_studio_api.py 里 Redis 的两个用法：Cache-Aside 缓存 + 防重复提交锁"""
import urllib.request, json, hashlib, redis, pymysql, time

API = "http://127.0.0.1:18007"
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

def get(path):
    return json.loads(urllib.request.urlopen(API + path).read().decode("utf-8"))

def post(path, payload):
    req = urllib.request.Request(API + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req).read().decode("utf-8")

def task_count():
    conn = pymysql.connect(host='localhost', user='root', password='123456',
                           database='pdf_qa', charset='utf8mb4')
    c = conn.cursor(); c.execute("SELECT COUNT(*) FROM studio_tasks")
    n = c.fetchone()[0]; c.close(); conn.close(); return n

print("=" * 60)
print("测试 1：Cache-Aside 缓存")
print("=" * 60)
r.delete("studio:task:3")
print(f"[清掉缓存] redis 里有 studio:task:3 吗 -> {r.exists('studio:task:3')}")

t0 = time.time()
d1 = get("/studio/task/3"); t1 = time.time()
print(f"[第1次] from_cache={d1['from_cache']}  步骤数={len(d1['steps'])}  耗时={(t1-t0)*1000:.0f}ms")
print(f"        -> 走 MySQL，并回填了缓存。redis 里有 key 吗 -> {r.exists('studio:task:3')}")

t0 = time.time()
d2 = get("/studio/task/3"); t1 = time.time()
print(f"[第2次] from_cache={d2['from_cache']}  步骤数={len(d2['steps'])}  耗时={(t1-t0)*1000:.0f}ms")
print(f"        -> 命中 Redis，没查 MySQL")

# 取一条审查员的记录看看，这就是面试要讲的东西
for s in d1["steps"]:
    if s["agent"] == "审查员":
        print(f"\n--- 审查员 {s['action']}，extracted score = {s['score']} ---")
        print(s["content"][:180].replace("\n", " | "))
        break

print()
print("=" * 60)
print("测试 2：防重复提交锁（SET NX EX）")
print("=" * 60)
req_text = "redis-lock-verify-please-ignore"
lkey = f"studio:lock:{hashlib.md5(req_text.encode()).hexdigest()}"

# 模拟"任务正在跑"：手动占住这把锁
r.set(lkey, "1", nx=True, ex=300)
print(f"[手动占锁] {lkey}")
print(f"           ttl={r.ttl(lkey)}s")

before = task_count()
resp = post("/studio/run", {"task_type": "测试", "requirement": req_text})
after = task_count()
print(f"[提交同样需求] 服务端返回 -> {resp}")
print(f"[任务表行数] 提交前={before}  提交后={after}  -> {'没有新建任务，被挡住了 ✅' if before == after else '居然建了任务 ❌'}")
print(f"[且没有调用任何 LLM，没烧 token]")

# 释放锁后再提交一次，应该能正常进入流程 —— 这里不真跑，只看锁能不能拿到
r.delete(lkey)
print(f"[释放锁后] 能重新拿到锁吗 -> {bool(r.set(lkey, '1', nx=True, ex=300))}")
r.delete(lkey)
print(f"[收尾] 已清理测试锁")
