"""
Transformer 核心：Self-Attention 的最小实现
用最简单的例子展示 "I love you" 里每个词怎么互相"关注"
"""

import numpy as np

# 1. 假设我们有 3 个词，每个词用 4 维向量表示
#    真实模型里是几百上千维，这里是简化
words = ["I", "love", "you"]
X = np.array([
    [0.2, 0.1, 0.3, 0.4],    # "I" 的向量
    [0.5, 0.3, 0.2, 0.1],    # "love" 的向量
    [0.1, 0.4, 0.3, 0.2],    # "you" 的向量
])

print("=" * 50)
print("输入：句子", words)
print("每个词的向量（简化）：")
print(X)
print()

# 2. Self-Attention 的简化版本（没有训练好的权重，直接用 X 本身）
#    实际还会乘上 Wq, Wk, Wv 三个权重矩阵来学
Q = X  # Query
K = X  # Key
V = X  # Value

# 3. 计算注意力分数：Q × Kᵀ
#    点积越大 → 两个词越相关
scores = Q @ K.T  # @ 是矩阵乘法
print("注意力分数矩阵（行i → 词i关注词j的程度）：")
print(np.round(scores, 3))
print()

# 4. 缩放（让数字不要太大）
d_k = X.shape[1]  # 向量维度 = 4
scores = scores / np.sqrt(d_k)

# 5. Softmax：把分数转成概率（让每行加起来=1）
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

attention_weights = softmax(scores)
print("Attention 权重（每行加起来=1，表示关注度占比）：")
print(np.round(attention_weights, 3))
print()

# 6. 用权重加权求和得到输出
output = attention_weights @ V
print("输出（每个词融合了其他词信息后的新向量）：")
print(np.round(output, 3))
print()

# 7. 直观解读
print("=" * 50)
print("🌟 直观解读：")
for i, word in enumerate(words):
    # 找出这个词最关注的前两个词
    weights = attention_weights[i]
    top_indices = np.argsort(weights)[::-1][:2]
    top_words = [f"{words[j]}({weights[j]:.2f})" for j in top_indices]
    print(f"  「{word}」最关注：{', '.join(top_words)}")

print()
print("💡 关键理解：每个词看完所有词后，")
print('   带着"上下文信息"变成了一个新的向量')
print("   — 这就是 Transformer 能理解上下文的原因！")
