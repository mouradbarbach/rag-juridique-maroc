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

st.set_page_config(page_title="المساعد القانوني", page_icon="⚖️", layout="centered")

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# ==========================================
# 1. Sawt dyal Salma (B tariqa m-securisé l Streamlit)
# ==========================================
async def _generate_audio_async(text):
    clean_text = text.replace("*", "").replace("#", "").replace("_", "")
    communicate = edge_tts.Communicate(clean_text, "ar-MA-SalmaNeural")
    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    await communicate.save(tmp_path)
    return tmp_path

def generate_audio_sync(text):
    # Had l-9aleb kay-fok l-mochkil dyal asyncio f Streamlit
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(_generate_audio_async(text))

# ==========================================
# 2. RAG Pipeline
# ==========================================
@st.cache_resource
def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    system_prompt = """
    أنت مساعد قانوني مغربي خبير. 
    استخدم المعلومات القانونية التالية للإجابة. إذا لم تجد الإجابة، قل "عذراً، لم أجد هذه المعلومة".
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
# 3. L-Interface Web
# ==========================================
st.markdown("<h1 style='text-align:center;'>⚖️ المساعد القانوني المغربي</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; direction:rtl;'>مرحبا بك! كيفاش نقدر نعاونك في القانون المغربي؟</p>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Afichi l-historique b HTML RTL
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f"<div dir='rtl' style='text-align:right; font-size:18px;'>{msg['content']}</div>", unsafe_allow_html=True)

if question := st.chat_input("سولني على القانون..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(f"<div dir='rtl' style='text-align:right; font-size:18px;'>{question}</div>", unsafe_allow_html=True)
        
    with st.chat_message("assistant"):
        with st.spinner("جاري البحث في القوانين..."):
            response = rag_chain.invoke({"input": question, "chat_history": st.session_state.chat_history})
            answer = response["answer"]
            st.markdown(f"<div dir='rtl' style='text-align:right; font-size:18px;'>{answer}</div>", unsafe_allow_html=True)
            
        with st.spinner("جاري توليد الصوت 🔊..."):
            audio_path = generate_audio_sync(answer)
            st.audio(audio_path, format="audio/mp3")
                
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.extend([("human", question), ("ai", answer)])