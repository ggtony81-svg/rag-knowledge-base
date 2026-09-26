"""
RAG 优化前后对比测试
同一个问题，分别用简单检索和重排序检索，对比结果
"""
import os, numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer, CrossEncoder

print("加载模型...")
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
reranker = CrossEncoder("BAAI/bge-reranker-base")

# ============================================================
# 模拟一个知识库（含有相似内容，会混淆检索）
# ============================================================
docs = [
    "第一条 年假规定：入职满一年的员工，可享受5天带薪年假。",
    "第二条 年假申请：年假需提前3个工作日通过OA系统申请。",
    "第三条 年假顺延：当年未休完的年假可顺延至次年3月底。",
    "第四条 病假规定：病假需提供医院证明，工资按80%发放。",
    "第五条 事假规定：事假全年不超过15天，无薪。",
    "第六条 加班费：加班按1.5倍工资计算加班费。",
    "第七条 法定节假日加班：法定节假日加班按3倍工资计算。",
    "第八条 迟到处理：迟到超30分钟记为半天事假。",
]

doc_vectors = model.encode(docs)

# ============================================================
# 测试问题（模糊提问，容易混淆）
# ============================================================
questions = [
    "加班怎么算钱",           # 有"1.5倍"和"3倍"两个相关段落
    "请假扣钱吗",             # 可能匹配到年假、病假、事假
    "年假能留着明年用吗",     # 具体问题，看能不能匹配到"顺延"
]

for q in questions:
    print("\n" + "=" * 60)
    print(f"🙋 问题：{q}")
    print("=" * 60)

    qv = model.encode([q])

    # ── 方法1：简单取 Top-3 ──
    scores = [np.dot(qv[0], dv) / (np.linalg.norm(qv[0]) * np.linalg.norm(dv)) for dv in doc_vectors]
    simple_top3 = np.argsort(scores)[-3:][::-1]

    # ── 方法2：先取 Top-5，再 Rerank ──
    top5 = np.argsort(scores)[-5:][::-1]
    pairs = [(q, docs[i]) for i in top5]
    re_scores = reranker.predict(pairs)
    rerank_top3 = [top5[i] for i in np.argsort(re_scores)[-3:][::-1]]

    # ── 并排打印 ──
    print()
    print(f"  方法               Top-1 内容")
    print(f"  {'─'*50}")
    print(f"  ① 简单向量检索      {docs[simple_top3[0]][:40]}")
    print(f"  ② 向量+Rerank       {docs[rerank_top3[0]][:40]}")
    print()
    print(f"  完整对比：")
    print(f"  【简单检索 Top-3】")
    for i, idx in enumerate(simple_top3):
        print(f"    {i+1}. [相关度 {scores[idx]:.2f}] {docs[idx]}")
    print(f"  【Rerank 精排 Top-3】")
    for i, idx in enumerate(rerank_top3):
        rs = re_scores[list(top5).index(idx)]
        print(f"    {i+1}. [Rerank分 {rs:.2f}] {docs[idx]}")
    print()

print("=" * 60)
print("说明：Rerank 会给每个(问题,文章)对单独打分")
print("能更好地区分『看起来相关』和『真正相关』")
print("比如『年假顺延』这条，简单检索可能排不上去")
print("但 Rerank 能识别出它跟『留着明年用』最相关")
