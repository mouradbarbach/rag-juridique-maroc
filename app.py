import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Maktabat jdad bach n-gaddou l'3arbiya f l'terminal
import arabic_reshaper
from bidi.algorithm import get_display

def fix_arabic(text):
    """Fonction bach t-lsse9 l'7ourouf w t-9leb l'ktaba mn Liser l-Limen"""
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return bidi_text

# 1. Njbdou l'Key dyal GitHub
load_dotenv()
github_token = os.getenv("GITHUB_TOKEN")

if not github_token:
    print("❌ L'Token dyal GitHub makaynch f .env!")
    exit()

# 2. n-chargiw l'Base de données FAISS w l'Embeddings
print("🧠 Kan-fiye9ou l'Base de données l'9anouniya...")
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 3. N-gadou l'LLM (GPT-4o L-Mjehed)
print("🤖 Kan-rbtou m3a GPT-4o L-Mjehed...")
llm = ChatOpenAI(
    api_key=github_token,
    base_url="https://models.inference.ai.azure.com", 
    model="gpt-4o", 
    temperature=0.1 
)

# 4. L'Prompt jdid m3a ta3limat d-logha
system_prompt = (
    "أنت مساعد قانوني مغربي ذكي ومحترف. "
    "قاعدة هامة جداً: أجب دائماً بنفس اللغة أو اللهجة التي طُرح بها السؤال. "
    "إذا سألك المواطن بالدارجة المغربية، أجب بالدارجة المغربية المبسطة والمفهومة. "
    "إذا سألك بالفرنسية، أجب بالفرنسية. وإذا سألك بالعربية الفصحى، أجب بالفصحى.\n\n"
    "استخدم فقط المعلومات القانونية التالية للإجابة على سؤال المواطن. "
    "إذا لم تجد الإجابة في هذه المعلومات، قل 'عذراً، لم أجد هذه المعلومة في النصوص القانونية المتاحة' بنفس لغة السؤال، ولا تخترع أي معلومات من عندك.\n\n"
    "المعلومات القانونية:\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# 5. N-jm3ou l'Pipeline (RAG)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 6. Chat Loop (L'Interface f l'terminal)
print("\n" + "="*50)
print("⚖️  L'Assistant L'9anouni L'Maghribi Wajd! (Ktab 'q' bach tkhrej)")
print("="*50 + "\n")

while True:
    question = input("👤 L'Mowatin: ")
    if question.lower() == 'q':
        print("👋 Bslama! L'Assistant kay-sed l'bureau dyalo.")
        break
    
    print("⏳ Kay-9leb f l'9anoun...")
    response = rag_chain.invoke({"input": question})
    
    # KAN-GADDOU L'AFFICHAGE DYAL L'JAWAB
    fixed_answer = fix_arabic(response['answer'])
    
    print(f"\n⚖️  L'Assistant:\n{fixed_answer}")
    print("\n" + "-"*50 + "\n")