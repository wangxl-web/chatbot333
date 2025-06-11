import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import os


# 初始化模型和知识库
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 示例文献段落（你可替换成更大语料）
documents = [
    "肥胖是一种慢性代谢性疾病。",
    "儿童青少年肥胖率近年来显著上升，已成为重要公共卫生问题。",
    "膳食结构不合理、饮食行为不健康是造成肥胖的重要原因。",
    "儿童肥胖会影响心理健康、认知能力、呼吸系统及心血管系统。",
    "中医认为肥胖与痰湿、脾虚、情志所伤等因素有关。",
]
# 编码并建立向量索引
embeddings = model.encode(documents)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# DeepSeek API
API_KEY = st.secrets["DEEPSEEK_API_KEY"]

def retrieve_docs(question, top_k=3):
    q_embed = model.encode([question])
    D, I = index.search(q_embed, top_k)
    return [documents[i] for i in I[0]]

def call_deepseek_api(question, context):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个营养学专家，请结合提供的资料回答用户问题。"},
            {"role": "user", "content": f"问题：{question}\n相关文献：{context}"}
        ],
        "temperature": 0.7
    }

    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    result = response.json()
    return result.get("choices", [{}])[0].get("message", {}).get("content", "未能生成回答。")

# --- Streamlit 页面布局 ---
st.set_page_config(page_title="营养知识机器人", layout="centered")
st.title("🥗 营养知识问答机器人")
st.markdown("请输入您的问题，我会结合营养知识库为您解答：")

question = st.text_area("您的问题", placeholder="例如：青少年为什么容易肥胖？")
if st.button("提交问题") and question.strip():
    with st.spinner("生成回答中，请稍候..."):
        relevant = retrieve_docs(question)
        context = "\n".join(relevant)
        answer = call_deepseek_api(question, context)
        st.markdown("### 🤖 回答如下：")
        st.write(answer)
