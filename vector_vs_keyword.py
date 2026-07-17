"""
向量数据库 Chroma 实操演示
对比关键词搜索 vs 向量搜索的差距
"""

# ============================================================
# 第1部分：准备数据
# ============================================================
documents = [
    "入职满一年员工可享受5天年假，满10年可享受10天年假",
    "年假需提前3个工作日通过OA系统申请",
    "春节放假7天，国庆放假7天，元旦放假3天",
    "病假需提供二级以上医院开具的证明",
    "事假全年累计不得超过15天，需部门主管审批",
    "薪资每月15日发放，遇节假日提前",
    "加班可申请调休或按1.5倍工资计算加班费",
]

# ============================================================
# 第2部分：关键词搜索（旧方式）
# ============================================================
def keyword_search(query):
    """找包含问题里任意关键词的文档"""
    # 去掉无意义的字
    stop_words = "的了是吗？?！!"
    keywords = query
    for w in stop_words:
        keywords = keywords.replace(w, "")
    results = []
    for doc in documents:
        score = sum(1 for kw in keywords if kw in doc)
        if score > 0:
            results.append((doc, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:2]

# ============================================================
# 第3部分：向量搜索（新方式）
# ============================================================
# 用轻量级 embedding：所有文档的向量预先算好，存成列表
# 后续用余弦相似度来搜

def get_embedding(text):
    """
    这是一段模拟的 embedding 函数。
    真实情况调 sentence-transformers 模型。

    但这节课我们先理解流程区别，
    所以用"预定义的相似度"代替真正的模型计算。
    """
    pass  # 下面测试用例直接给相似度

# 预先定义：每个问题跟每条文档的"真实语义相似度"
# （模拟向量搜索的效果）
vector_search_results = {
    "请假的流程是什么": [
        ("年假需提前3个工作日通过OA系统申请", 0.91),
        ("事假全年累计不得超过15天，需部门主管审批", 0.82),
    ],
    "我想出去旅游几天": [
        ("入职满一年员工可享受5天年假，满10年可享受10天年假", 0.89),
        ("事假全年累计不得超过15天，需部门主管审批", 0.85),
    ],
    "发工资了没有": [
        ("薪资每月15日发放，遇节假日提前", 0.95),
        ("加班可申请调休或按1.5倍工资计算加班费", 0.30),
    ],
    "节日休息几天": [
        ("春节放假7天，国庆放假7天，元旦放假3天", 0.96),
        ("年假需提前3个工作日通过OA系统申请", 0.25),
    ],
}

test_questions = [
    ("请假的流程是什么", "想请假"),
    ("我想出去旅游几天", "旅游出去玩"),
    ("发工资了没有", "工资发放"),
    ("节日休息几天", "放假"),
]

print("=" * 70)
print("对比：关键词搜索 vs 向量搜索")
print("=" * 70)
print()

for question, keyword_query in test_questions:
    print("-" * 70)
    print(f"🙋 用户问：『{question}』")
    print()

    # 关键词搜索
    kw_results = keyword_search(keyword_query)
    print(f"🔍 关键词搜索（搜『{keyword_query}』）→ 找到 {len(kw_results)} 条:")
    if kw_results:
        for doc, score in kw_results:
            print(f"    匹配 {score} 个关键词 → {doc}")
    else:
        print(f"    ❌ 没找到！因为『请假』里没有『旅游』『工资』这些字")
    print()

    # 向量搜索
    vs_results = vector_search_results[question]
    print(f"📊 向量搜索（按意思找）→ 找到 {len(vs_results)} 条:")
    for doc, score in vs_results:
        print(f"    相似度 {score:.0%} → {doc}")
    print()

print("=" * 70)
print("📌 关键对比")
print()
print("关键词搜索只认『字』:")
print("  用户问『请假流程』→ 搜『请假』→ 找到")
print("  用户问『想出去旅游』→ 搜『旅游』→ 找不到！『年假』里没有『旅游』二字")
print()
print("向量搜索认『意思』:")
print("  『旅游』和『年假』意思相近（都是休假）")
print("  『发工资』和『薪资』意思相近")
print("  向量模型懂这些，所以能命中")
print()
print("✅ 这就是为什么企业级 RAG 必须用向量数据库")
print("   靠关键词搜索，换个说法就找不到了")
