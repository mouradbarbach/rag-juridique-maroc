import os
import asyncio
import tempfile
import streamlit as st
import edge_tts
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

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# ==========================================
# 2. Fonction dyal Sawt (Salma) M9adda
# ==========================================
def generate_audio_sync(text):
    # N-7iydou r-romouz li kay-khab9ou s-sawt d Salma
    clean_text = text.replace("*", "").replace("#", "").replace("_", "")
    communicate = edge_tts.Communicate(clean_text, "ar-MA-SalmaNeural")
    
    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    
    try:
        asyncio.run(communicate.save(tmp_path))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(communicate.save(tmp_path))
        
    return tmp_path

# ==========================================
# 3. RAG Pipeline
# ==========================================
@st.cache_resource
def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
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
# 4. L-Interface w l-Chat Loop
# ==========================================
st.markdown("<h1>⚖️ المساعد القانوني المغربي</h1>", unsafe_allow_html=True)
st.markdown("<p>مرحبا بك! كيفاش نقدر نعاونك في القانون المغربي؟</p>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("سولني على القانون..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
        
    with st.chat_message("assistant"):
        with st.spinner("جاري البحث في القوانين..."):
            response = rag_chain.invoke({
                "input": question, 
                "chat_history": st.session_state.chat_history
            })
            answer = response["answer"]
            st.write(answer)
            
        with st.spinner("جاري توليد الصوت 🔊..."):
            try:
                audio_path = generate_audio_sync(answer)
                st.audio(audio_path, format="audio/mp3")
            except Exception as e:
                st.error("تعذر تشغيل الصوت حالياً.")
                
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.extend([("human", question), ("ai", answer)])