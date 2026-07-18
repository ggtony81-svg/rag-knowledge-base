"""
PDF 问答全流程可视化演示
让你看到：上传 → 解析 → 分块 → 向量化 → 检索 → 回答 每一步的效果
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
import numpy as np

print("=" * 70)
print("📋 PDF 问答全流程演示")
print("=" * 70)
print()

# ============================================================
# 第1步：准备 PDF 内容（模拟从 PDF 提取的原始文本）
# ============================================================
print("第1步 📄 PDF 原始内容（模拟员工手册）")
print("-" * 70)

raw_text = """公司员工手册（2026版）

第一章 休假制度

第一条 年假
入职满一年的员工，可享受5天带薪年假。
入职满十年的员工，可享受10天带薪年假。
年假需提前3个工作日通过OA系统申请。

第二条 病假
员工请病假需提供二级以上医院开具的证明。
病假期间工资按基本工资的80%发放。

第三条 事假
事假全年累计不得超过15天，需部门主管审批。
事假为无薪假。

第二章 薪酬福利

第四条 薪资发放
薪资每月15日发放，遇节假日提前至前一个工作日。

第五条 加班费
加班可申请调休或按1.5倍工资计算加班费。
法定节假日加班按3倍工资计算。

第三章 考勤制度

第六条 迟到早退
迟到超过30分钟记为半天事假。
早退需部门主管审批。

第七条 法定节假日
春节放假7天，国庆放假7天，元旦放假3天。
"""

print(raw_text)
print(f"📊 原始文本长度：{len(raw_text)} 字符")
print()

# ============================================================
# 第2步：分块
# ============================================================
print("第2步 ✂️ 分块结果（按章节切分成 7 块）")
print("-" * 70)

lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
chunks = []
current = ""
for line in lines:
    if line.startswith(("第", "公司")):
        if current:
            chunks.append(current.strip())
        current = line + "\n"
    else:
        current += line + "\n"
if current:
    chunks.append(current.strip())

for i, chunk in enumerate(chunks):
    print(f"  块{i+1} ({len(chunk):3d}字符): {chunk[:50].replace(chr(10),' ')}...")

print(f"\n📊 共 {len(chunks)} 块，平均每块 {sum(len(c) for c in chunks)//len(chunks)} 字符")
print()

# ============================================================
# 第3步：向量化
# ============================================================
print("第3步 🔢 向量化（每个块 → 384维向量）")
print("-" * 70)

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
vectors = model.encode(chunks)

for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
    print(f"  块{i+1}: [{', '.join(f'{v:.2f}' for v in vec[:4])}, ...] ← 384个数字")

print(f"\n📊 向量维度: {vectors.shape[1]}D")
print()

# ============================================================
# 第4步：检索演示
# ============================================================
print("第4步 🎯 检索演示（用户提问 → 找最相关的块）")
print("-" * 70)

questions = [
    "请假怎么请",
    "加班有没有加班费",
    "过年放几天假",
    "我迟到了会怎么样"
]

for q in questions:
    q_vec = model.encode([q])
    scores = []
    for d_vec in vectors:
        cos_sim = np.dot(q_vec[0], d_vec) / (np.linalg.norm(q_vec[0]) * np.linalg.norm(d_vec))
        scores.append(cos_sim)

    top_idx = np.argsort(scores)[-1:][::-1][0]

    print(f"  用户问: 「{q}」")
    print(f"  最匹配: 块{top_idx+1}（相似度 {scores[top_idx]:.2f}）")
    print(f"  内容: {chunks[top_idx][:60].replace(chr(10),' ')}")
    print()

# ============================================================
# 第5步：最终效果
# ============================================================
print("第5步 🤖 最终：把检索到的块 + 问题 → 发给 DeepSeek → 得到准确回答")
print("-" * 70)
print("""
用户：请假怎么请？

① 检索到最相关的块：
   块1「第一条 年假...」「第三条 事假...」

② 拼成 Prompt：
   资料：第一条 年假... 第三条 事假...
   问题：请假怎么请？

③ 发给 DeepSeek → 回答：
   根据公司制度，年假需提前3个工作日申请，
   事假需部门主管审批，全年累计不超过15天。
""")
