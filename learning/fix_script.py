# 清理损坏的行
with open('d:/shuqi/agent_studio_api.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 删除损坏的孤立行（127-128, 132-133 附近）
clean = []
skip_next = False
for i, line in enumerate(lines):
    stripped = line.strip()
    # 跳过损坏的行：孤立的中文 f-string 碎片
    if stripped in ['成果：', '{work}"', '上一版成果：', '{prev_work}"']:
        continue
    clean.append(line)

with open('d:/shuqi/agent_studio_api.py', 'w', encoding='utf-8') as f:
    f.writelines(clean)

print('cleaned')
