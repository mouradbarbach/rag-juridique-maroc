import os
import re
import asyncio
import tempfile
import edge_tts
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

async def text_to_speech(text):
    try:
        clean_text = re.sub(r'[*#_`~]', '', text)
        voice = "ar-MA-SalmaNeural"
        communicate = edge_tts.Communicate(clean_text, voice)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_path = tmp_file.name
        await communicate.save(tmp_path)
        return tmp_path
    except Exception as e:
        print(f"❌ Erreur f s-sawt: {e}")
        return None

st.set_page_config(page_title="المساعد القانوني الذكي", page_icon="⚖️")
st.markdown("<style>body, .stChatMessage, .stMarkdown { direction: rtl; text-align: right; } .stChatInputContainer { direction: ltr; }</style>", unsafe_allow_html=True)
st.title("⚖️ المساعد القانوني المغربي الصوتي")

@st.cache_resource
def load_assistant():
    load_dotenv()
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatOpenAI(
        api_key=os.environ.get("GITHUB_TOKEN"),
        base_url="https://models.inference.ai.azure.com", 
        model="gpt-4o",
        temperature=0.1 
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "قم بصياغة سؤال مستقل يمكن فهمه دون سجل الدردشة. لا تجب على السؤال، فقط أعد صياغته."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
        "أنت مساعد قانوني مغربي ذكي ومحترف. أجب دائماً بنفس اللغة أو اللهجة التي طُرح بها السؤال.\n"
        "استخدم فقط المعلومات القانونية التالية للإجابة. إذا لم تجد الإجابة، قل 'عذراً، لم أجد هذه المعلومة في النصوص القانونية المتاحة' ولا تخترع أي معلومات.\n\n"
        "المعلومات القانونية:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

rag_chain = load_assistant()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 
if "messages" not in st.session_state:
    st.session_state.messages = [] 

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("بماذا يمكنني مساعدتك اليوم؟ (اكتب سؤالك)"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("⏳ أتفحص القوانين..."):
            response = rag_chain.invoke({"input": question, "chat_history": st.session_state.chat_history})
            answer = response["answer"]
            st.markdown(answer)
            
        with st.spinner("🔊 جاري توليد الصوت..."):
            audio_path = asyncio.run(text_to_speech(answer))
            if audio_path:
                st.audio(audio_path, format="audio/mp3")
            else:
                st.warning("⚠️ السيرفر الصوتي عليه ضغط حالياً.")
            
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.extend([("human", question), ("ai", answer)])