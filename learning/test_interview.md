# AI 面试官测试用例

## 岗位 JD（贴到左边框）

AI 大模型应用开发工程师

岗位职责：
1. 负责公司 AI 产品的后端架构设计与开发
2. 参与大模型应用落地，包括 RAG 系统、Agent 工具调用等
3. 对接大模型 API（DeepSeek/Qwen 等），实现流式输出
4. 优化系统性能，保障服务稳定性
5. 使用 MySQL、Redis 存储业务数据并优化查询

任职要求：
- 熟练掌握 Python，有 FastAPI/Flask 开发经验
- 熟悉大模型 API 调用，了解 RAG 检索增强生成原理
- 熟悉 MySQL、Redis
- 有 AI 应用落地项目经验优先
- 良好的代码习惯和团队协作能力

---

## 测试简历（贴到右边框）

张三 | 138-0000-0000 | zhangsan@email.com | 3年经验

工作经历：

ABC科技有限公司 | 后端开发工程师 | 2022.06-至今
- 使用 Python + FastAPI 开发 RESTful API，支撑日均 10 万+请求
- 搭建 RAG 知识库问答系统，集成 DeepSeek API 实现流式回答
- 使用 MySQL 存储对话历史，Redis 缓存加速，接口响应降低 40%
- 优化数据库索引，解决慢查询问题

项目经验：

AI 知识库问答系统（个人项目）
- 基于 FastAPI + Sentence-Transformers 向量检索
- 集成 CrossEncoder 重排序，检索准确率提升
- 支持 PDF 上传、自动分块、向量化、流式问答
- 接入 MySQL 持久化 + Redis 缓存 + Streamlit 前端

技能：
- Python / FastAPI / Streamlit
- DeepSeek API / 流式输出
- MySQL / Redis
- RAG / 向量检索

---

## 使用方法

1. 先启动 API：`python interview_api.py`
2. 再启动前端：`streamlit run interview_app.py`
3. 打开 `http://localhost:8501`
4. 点「开始面试」，粘贴上面两份内容
5. 点「开始面试」→ 看 AI 面试官的开场
6. 试着回答，看 AI 怎么追问和点评
