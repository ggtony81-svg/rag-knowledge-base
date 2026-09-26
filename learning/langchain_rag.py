"""
LangChain 完整 RAG 演示（对比你手写的 pdf_qa_redis.py）
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from config import DEEPSEEK_API_KEY
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ============================================================
# 模拟知识库
# ============================================================
documents = [
    "入职满一年员工可享受5天年假，满10年可享受10天年假",
    "年假需提前3个工作日通过OA系统申请",
    "加班可申请调休或按1.5倍工资计算加班费",
    "薪资每月15日发放，遇节假日提前",
    "病假需提供二级以上医院开具的证明",
]

print("=" * 60)
print("LangChain RAG 完整链路")
print("=" * 60)

# ============================================================
# 1. Embedding 模型
# ============================================================
print("\n1. 加载 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# ============================================================
# 2. 向量库
# ============================================================
print("2. 构建向量库...")
vectorstore = Chroma.from_documents(
    [Document(page_content=d) for d in documents],
    embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ============================================================
# 3. 大模型（DeepSeek 兼容 OpenAI 格式）
# ============================================================
print("3. 连接 DeepSeek...")
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1",
    temperature=0.7
)

# ============================================================
# 4. Prompt 模板
# ============================================================
prompt = ChatPromptTemplate.from_messages([
    ("system", "基于以下资料回答问题，不知道就说不知道：\n{context}"),
    ("human", "{input}")
])

# ============================================================
# 5. RAG Chain
# ============================================================
print("4. 构建 RAG Chain...")
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# ============================================================
# 6. 测试
# ============================================================
test_questions = [
    "加班有没有加班费",
    "年假怎么申请",
    "工资什么时候发",
]

for q in test_questions:
    print(f"\n问题：{q}")
    result = rag_chain.invoke({"input": q})
    print(f"回答：{result['answer']}")
    print(f"参考文档：{[d.page_content[:30] for d in result['context']]}")

print("\n" + "=" * 60)
print("对比你手写的 pdf_qa_redis.py：")
print("  你手写：200+ 行代码，自己处理向量化、检索、拼接 prompt")
print("  LangChain：50 行搞定，核心逻辑就是搭积木")
print("=" * 60)
