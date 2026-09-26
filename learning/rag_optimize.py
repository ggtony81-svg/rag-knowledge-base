"""
RAG 优化演示
1. 分块策略对比
2. 重排序（Rerank）
3. 检索效果评估
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
import numpy as np

print("=" * 60)
print("RAG 优化技巧")
print("=" * 60)

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# ============================================================
# 模拟文档
# ============================================================
doc_text = """
公司员工手册（2026版）

第一章 休假制度

第一条 年假规定
入职满一年的员工，可享受5天带薪年假。
入职满十年的员工，可享受10天带薪年假。
年假需提前3个工作日通过OA系统申请，经部门主管审批后方可休假。
年假可以拆分使用，但每次最少休半天。
当年未休完的年假可以顺延至次年3月底前使用。

第二条 病假管理
员工请病假需提供二级以上医院开具的证明。
病假期间工资按基本工资的80%发放。
病假超过30天的，需经人力资源部审批。

第三条 事假规定
事假全年累计不得超过15天，需部门主管审批。
事假为无薪假。

第二章 薪酬福利

第四条 薪资发放
薪资每月15日发放，遇节假日提前至前一个工作日。
薪资明细可通过OA系统查看。

第五条 加班费规定
加班可申请调休或按1.5倍工资计算加班费。
法定节假日加班按3倍工资计算。
加班需提前在OA系统提交申请，经审批后方可计算加班费。

第三章 考勤制度

第六条 迟到早退处理
迟到超过30分钟记为半天事假。
早退需部门主管审批。

第七条 法定节假日
春节放假7天，国庆放假7天，元旦放假3天。
"""

# ============================================================
# 策略1：简单分块（你现在的方式）
# ============================================================
print("\n策略1 📦 简单分块（按标点符号切）")
print("-" * 40)

def simple_chunk(text, chunk_size=100):
    """简单按字符数切块"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

chunks1 = simple_chunk(doc_text, 100)
# 去掉过短的
chunks1 = [c for c in chunks1 if len(c) > 20]
print(f"  共 {len(chunks1)} 块")
print(f"  平均每块 {sum(len(c) for c in chunks1)//len(chunks1)} 字符")
print(f"  问题：可能把『年假规定』和『年假申请』切成两块")
print()

# ============================================================
# 策略2：语义分块（优化后）
# ============================================================
print("策略2 🎯 语义分块（按章节/条款切块 + 重叠）")
print("-" * 40)

def semantic_chunk(text, chunk_size=150, overlap=30):
    """
    按章节标题分块，每块保留上下文重叠
    overlap=重叠字符数（保证跨段信息不丢失）
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    chunks = []
    current_chunk = ""

    for line in lines:
        # 遇到章节标题开始新块
        if any(line.startswith(p) for p in ["第一章", "第二章", "第三章", "第一条", "第二条", "第三条", "第四条", "第五条", "第六条", "第七条"]):
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    # 添加重叠：前一块的最后30个字拼到下一块开头
    final_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0 and overlap > 0:
            # 从前一块尾部取 overlap 个字
            prev_tail = chunks[i-1][-overlap:]
            chunk = prev_tail + "\n" + chunk
        final_chunks.append(chunk)

    return final_chunks

chunks2 = semantic_chunk(doc_text, overlap=30)
print(f"  共 {len(chunks2)} 块")
for i, c in enumerate(chunks2):
    print(f"  块{i+1}: {c[:60].replace(chr(10),' ')}...")
print()

# ============================================================
# 对比：同样的问题，两种分块的检索效果
# ============================================================
print("=" * 60)
print("对比测试：同样的问题 → 不同分块 → 检索结果不同")
print("=" * 60)

questions = [
    "年假能拆分休吗",
    "加班费怎么算",
    "病假工资怎么发",
]

for q in questions:
    print(f"\n问题：「{q}」")
    q_vec = model.encode([q])

    # 策略1检索
    c1_vec = model.encode(chunks1)
    scores1 = [np.dot(q_vec[0], v) / (np.linalg.norm(q_vec[0]) * np.linalg.norm(v)) for v in c1_vec]
    top1 = np.argmax(scores1)

    # 策略2检索
    c2_vec = model.encode(chunks2)
    scores2 = [np.dot(q_vec[0], v) / (np.linalg.norm(q_vec[0]) * np.linalg.norm(v)) for v in c2_vec]
    top2 = np.argmax(scores2)

    print(f"  简单分块 → 块{top1+1}: {chunks1[top1][:60].replace(chr(10),' ')}...")
    print(f"  语义分块 → 块{top2+1}: {chunks2[top2][:60].replace(chr(10),' ')}...")
    print(f"  ✅ 语义分块信息更完整" if len(chunks2[top2]) > len(chunks1[top1]) else f"  ⚠️ 两者差不多")

print()
print("=" * 60)
print("优化前后对比")
print("=" * 60)
print(f"""
  优化前（简单分块）:
    - 信息被切碎，『年假』相关的内容可能散落在多个块里
    - 检索可能只找到部分信息

  优化后（语义分块 + 重叠）:
    - 按逻辑分组（按章/条切），信息完整
    - 重叠保留上下文，跨段信息不丢失
    - 检索命中率更高

  下一层优化：重排序（Rerank）
    - 检索出 Top-5 后，用更精确的模型重新打分
    - 把最相关的结果排在最前面
""")

# 重排序原理
print("=" * 60)
print("重排序（Rerank）原理")
print("=" * 60)
print("""
  向量检索阶段（粗筛）:
    问题 → 跟所有块算相似度 → 取 Top-5
    快，但不精确

  重排序阶段（精排）:
    Top-5 块 + 问题 → 逐个算相关性分数 → 重新排序
    慢一点，但更准

  比喻：
    向量检索 = 海选（5 万人选 100 人）
    重排序  = 复试（100 人选 Top-3）
""")
