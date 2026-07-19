"""
MySQL 实操：建表 → 存数据 → 查数据
结合你的 PDF 问答场景：存储聊天记录
"""
import pymysql
from datetime import datetime

# ============================================================
# 第1步：连接 MySQL 并创建数据库
# ============================================================
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='123456',
    charset='utf8mb4'
)
cursor = conn.cursor()

# 创建数据库（如果不存在）
cursor.execute("CREATE DATABASE IF NOT EXISTS pdf_qa DEFAULT CHARACTER SET utf8mb4")
cursor.execute("USE pdf_qa")
print("✅ 数据库 pdf_qa 就绪")
print()

# ============================================================
# 第2步：创建表
# ============================================================
cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INT AUTO_INCREMENT PRIMARY KEY,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        filename VARCHAR(255),
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")
print("✅ 表 chat_history 就绪")

# 看看表结构
cursor.execute("DESCRIBE chat_history")
print("\n📋 表结构：")
for col in cursor.fetchall():
    print(f"   {col[0]:15s} {col[1]:20s}")
print()

# ============================================================
# 第3步：插入数据（模拟保存聊天记录）
# ============================================================
history = [
    ("年假有几天", "入职满一年可享受5天年假，满10年可享受10天年假。", "员工手册.pdf"),
    ("加班有钱吗", "加班可以申请调休或按1.5倍工资计算加班费。", "员工手册.pdf"),
    ("请假怎么请", "年假需提前3个工作日通过OA系统申请。", "员工手册.pdf"),
]

for q, a, f in history:
    cursor.execute(
        "INSERT INTO chat_history (question, answer, filename) VALUES (%s, %s, %s)",
        (q, a, f)
    )
conn.commit()
print(f"✅ 插入了 {len(history)} 条聊天记录")
print()

# ============================================================
# 第4步：查询数据
# ============================================================
print("📊 最近聊天记录：")
cursor.execute("SELECT id, question, filename, created_at FROM chat_history ORDER BY id DESC")
for row in cursor.fetchall():
    print(f"   #{row[0]} {row[1]:20s} | {row[2]:15s} | {row[3]}")

print()

# ============================================================
# 第5步：跟 PDF 问答结合起来
# ============================================================
print("💡 实际用途：")
print("   用户提问 → 先查 MySQL 有没有类似历史 → 有就直接返回历史")
print("   没有 → 调模型 → 把新问答存进 MySQL")
print("   → 下次相同的问法秒回，省 token 又省时间")
print()

# 清理
cursor.close()
conn.close()
print("✅ MySQL 连接已关闭")
