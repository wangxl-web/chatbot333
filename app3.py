import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests

# 加载向量文件
with np.load("embeddings.npz", allow_pickle=True) as data:
    documents = data["documents"].tolist()
    embeddings = data["embeddings"]

# 加载模型
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 建立向量索引
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# API 设置
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = st.secrets["API_KEY"]  # 安全写法

# 检索函数
def retrieve_docs(question, top_k=3):
    q_embed = model.encode([question])
    D, I = index.search(q_embed, top_k)
    return [documents[i] for i in I[0]]

# API 调用函数
def call_deepseek_api(question, context):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个营养学专家，请结合资料回答用户问题。"},
            {"role": "user", "content": f"问题：{question}\n相关文献：{context}"}
        ],
        "temperature": 0.7
    }
    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    result = response.json()
    return result.get("choices", [{}])[0].get("message", {}).get("content", "未能生成回答。")

# Streamlit 页面
st.set_page_config(page_title="🥗 营养问答机器人", layout="centered")
st.title("🥗 营养知识问答机器人")
question = st.text_area("请输入您的问题", placeholder="例如：青少年为什么容易肥胖？")

if st.button("提交问题") and question.strip():
    with st.spinner("生成中，请稍候..."):
        relevant = retrieve_docs(question)
        context = "\n".join(relevant)
        answer = call_deepseek_api(question, context)
        st.markdown("### 🤖 回答如下：")
        st.write(answer)
