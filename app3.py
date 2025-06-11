import streamlit as st
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer  # ✅ 加回来，仅用于编码问题

# 加载向量文件
with np.load("embeddings.npz", allow_pickle=True) as data:
    documents = data["documents"].tolist()
    embeddings = data["embeddings"]

# 建立向量索引
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# ✅ 加载模型（仅用于编码问题）
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
model.to("cpu")  # 显式加载到 CPU，避免 cloud 报错

# API 设置
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = st.secrets["API_KEY"]

# 检索函数
def retrieve_docs(question, top_k=3):
    q_embed = model.encode([question])
    D, I = index.search(q_embed, top_k)
    return [documents[i] for i in I[0]]

# API 调用函数
def call_deepseek_api(question, context):
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个营养学专家。"},
            {"role": "user", "content": f"根据以下内容回答问题：{context}\n\n问题是：{question}"}
        ],
        "temperature": 0.7
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    print("返回状态码：", response.status_code)
    print("返回内容：", response.text)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"未能生成回答（状态码: {response.status_code}）"

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
