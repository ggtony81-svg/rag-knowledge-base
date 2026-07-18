"""
PDF 解析 + 分块 演示

核心逻辑两部分：
1. 从 PDF 提取文字（用 PyMuPDF）
2. 把长文本切分成小块（为 RAG 做准备）
"""

# ============================================================
# 演示分块逻辑（这是 RAG 的关键步骤）
# ============================================================

# 模拟从 PDF 提取出来的长文本
pdf_text = """
公司员工手册（2026版）

第一章 休假制度

第一条 年假
入职满一年的员工，可享受5天带薪年假。
入职满十年的员工，可享受10天带薪年假。
年假需提前3个工作日通过OA系统申请。

第二条 病假
员工请病假需提供二级以上医院开具的证明。

第三条 事假
事假全年累计不得超过15天，需部门主管审批。

第二章 薪酬福利

第四条 薪资发放
薪资每月15日发放，遇节假日提前。

第五条 加班费
加班可申请调休或按1.5倍工资计算加班费。
法定节假日加班按3倍工资计算。

第三章 考勤制度

第六条 法定节假日
春节放假7天，国庆放假7天，元旦放假3天。
"""

print("=" * 60)
print("第1步：从 PDF 提取文字")
print("=" * 60)
print(pdf_text[:100] + "...")
print()

# ============================================================
# 分块策略：按章节/条款切分
# ============================================================
print("=" * 60)
print("第2步：分块（为 RAG 向量化做准备）")
print("=" * 60)
print()

lines = [l.strip() for l in pdf_text.split("\n") if l.strip()]
chunks = []
current_chunk = ""

for line in lines:
    # 遇到"第x章"或"第x条"时，说明新的一段开始了
    if any(line.startswith(prefix) for prefix in ["第一章", "第二章", "第三章", "第一条", "第二条", "第三条", "第四条", "第五条", "第六条"]):
        if current_chunk:
            chunks.append(current_chunk.strip())
        current_chunk = line + "\n"
    else:
        current_chunk += line + "\n"

if current_chunk:
    chunks.append(current_chunk.strip())

for i, chunk in enumerate(chunks):
    print(f"📄 块 {i+1} ({len(chunk)} 字符):")
    print(f"   {chunk.replace(chr(10), chr(10)+'   ')}")
    print()

print(f"共切成 {len(chunks)} 个块")
print()

# ============================================================
# 展示：每个块 → 转成向量 → 存向量库
# ============================================================
print("=" * 60)
print("第3步：每个块 → 向量化 → 存知识库")
print("（就是你在 rag_api_v2.py 里已经做的事）")
print("=" * 60)
print()
print("每个块都跟之前的 7 条文档一样，")
print("用 sentence-transformers 转成向量，")
print("存到向量索引里。")
print()
print("以后用户提问 → 搜这些块 → 调模型回答")
print()

# ============================================================
# 实战：解析你的真实 PDF
# ============================================================
print("=" * 60)
print("试试解析你的真实 PDF：")
print("=" * 60)
print()

import os
# 找一个 PDF 文件
pdf_found = False
for root, dirs, files in os.walk("C:/Users/1"):
    for f in files:
        if f.endswith(".pdf"):
            pdf_path = os.path.join(root, f)
            pdf_found = True
            break
    if pdf_found:
        break

if pdf_found:
    print(f"找到一个 PDF: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        print(f"页数: {len(doc)}")
        # 只读第一页前500字
        text = doc[0].get_text()[:500]
        print(f"第一页预览: {text}")
        doc.close()
    except:
        print("（但可能受保护或不是文字型PDF，仅供参考）")
else:
    print("（没找到 PDF 文件，跳过）")

print()
print("💡 核心：PDF RAG 就比你现在的系统多两步")
print("   1. PDF 解析（提取文字）")
print("   2. 自动分块（切成小段）")
print("   后面的向量化、检索、生成 → 你已经有了")
