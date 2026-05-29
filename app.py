import os
import asyncio
import tempfile
import streamlit as st
import edge_tts
import subprocess
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# ==========================================
# 1. I3dadat d Sef7a w CSS bach l-3arabiya tban n9iya 100% (RTL)
# ==========================================
st.set_page_config(page_title="المساعد القانوني", page_icon="⚖️", layout="centered")
st.markdown("""
<style>
    .stMarkdown, p, div, h1, h2, h3, input {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Arial', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Jbed l-API Key dyal GitHub
# ==========================================
load_dotenv()
github_token = os.getenv("GITHUB_TOKEN", "")

# ==========================================
# 3. Fonction dyal Sawt (Salma) - M9adda l Streamlit
# ==========================================
async def _generate_audio_async(text):
    # N-n9iw l-ktaba mn ga3 r-romouz li kay-7ebsou Salma (njmat, 3awared, d-stour jdad...)
    clean_text = text.replace("*", "").replace("#", "").replace("_", "").replace("-", ".").replace("\n", " ")
    
    communicate = edge_tts.Communicate(clean_text, "ar-MA-SalmaNeural")
    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    await communicate.save(tmp_path)
    return tmp_path

def generate_audio_sync(text):
    # N-n9iw l-ktaba mn r-romouz w d-stour
    clean_text = text.replace("*", "").replace("#", "").replace("_", "").replace("-", ".").replace('"', '').replace("'", "")
    clean_text = " ".join(clean_text.splitlines())
    
    # N-creyiw fichier temporaire
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_path = tmp_file.name
    tmp_file.close()
    
    # N-lanciou edge-tts mn l-Console nichen (kay-t-jawez machakil Python)
    subprocess.run(['edge-tts', '--voice', 'ar-MA-SalmaNeural', '--text', clean_text, '--write-media', tmp_path])
    
    return tmp_path

# ==========================================
# 4. RAG Pipeline (B GPT-4o dyal GitHub Azure)
# ==========================================
@st.cache_resource
def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Hna m-gadda nichen m3a GitHub Student Pack
    llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0.3,
        api_key=github_token,
        base_url="https://models.inference.ai.azure.com"
    )
    
    system_prompt = """
    أنت مساعد قانوني مغربي خبير. 
    استخدم المعلومات القانونية التالية للإجابة.
    المعلومات:
    {context}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

rag_chain = load_rag_pipeline()

# ==========================================
# 5. L-Interface w l-Chat Loop
# ==========================================
st.markdown("<h1>⚖️ المساعد القانوني المغربي الصوتي</h1>", unsafe_allow_html=True)
st.markdown("<p>مرحبا بك! كيفاش نقدر نعاونك في القانون المغربي؟</p>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f"<div dir='rtl' style='text-align:right; font-size:18px;'>{message['content']}</div>", unsafe_allow_html=True)

if question := st.chat_input("سولني على القانون..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(f"<div dir='rtl' style='text-align:right; font-size:18px;'>{question}</div>", unsafe_allow_html=True)
        
    with st.chat_message("assistant"):
        with st.spinner("جاري البحث في القوانين..."):
            response = rag_chain.invoke({
                "input": question, 
                "chat_history": st.session_state.chat_history
            })
            answer = response["answer"]
            st.markdown(f"<div dir='rtl' style='text-align:right; font-size:18px;'>{answer}</div>", unsafe_allow_html=True)
            
        with st.spinner("جاري توليد الصوت 🔊..."):
            try:
                audio_path = generate_audio_sync(answer)
                st.audio(audio_path, format="audio/mp3")
            except Exception as e:
                st.error(f"تعذر تشغيل الصوت: {e}")
                
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.extend([("human", question), ("ai", answer)])