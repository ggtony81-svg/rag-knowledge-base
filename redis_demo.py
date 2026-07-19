"""
Redis 实操演示
在 PDF 问答里的实际用途：缓存问答结果
"""
import sys
import os

# 先把脚本目录从路径中去掉，避免 redis 文件夹干扰
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.remove(script_dir)

# 现在导入 redis（会从 site-packages 加载）
import redis

# 用完再加回来
sys.path.insert(0, script_dir)

# ============================================================
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
print("Redis 连接成功")
print()

# 第1步：基础操作
print("第1步：基础操作")
r.set("user:1:name", "小明")
print(f"  存入 → 取出: {r.get('user:1:name')}")
print()

# 第2步：模拟缓存问答
print("第2步：缓存问答结果（省 token 又省时间）")

def ask_with_cache(question):
    cache_key = f"qa:{question}"
    cached = r.get(cache_key)
    if cached:
        print(f"  缓存命中! 秒回: {cached}")
        return cached

    print(f"  缓存未命中 → 调 DeepSeek API...")
    answer = f"这是关于「{question}」的回答"
    r.setex(cache_key, 3600, answer)
    print(f"  已缓存，下次秒回")
    return answer

print(f"\n第一次提问：「年假有几天」")
print(f"  结果: {ask_with_cache('年假有几天')}")

print(f"\n第二次提问：「年假有几天」")
print(f"  结果: {ask_with_cache('年假有几天')}")
print()

# 第3步：查看状态
print(f"Redis 总 key 数: {r.dbsize()}")
r.close()
print("完成")
