import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

print("🧠 Chargement dyal l'Modèle d'Embedding (Multilingual)...")
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

print("📂 9raya dyal l'chunks mn JSON...")
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks_data = json.load(f)

documents = []
for item in chunks_data:
    doc = Document(
        page_content=item["text"], 
        metadata={"source": item["source"], "chunk_id": item["chunk_id"]}
    )
    documents.append(doc)

print(f"🚀 Génération dyal l'Vectors l {len(documents)} morceaux (hadchi ghay-akhoud chwiya dl wa9t)...")
vector_db = FAISS.from_documents(documents, embeddings)

print("💾 Sauvegarde dyal l'Base de données FAISS...")
vector_db.save_local("faiss_index")
print("🎉 Nadi! L'Base de données wajda w m-sauvegardya f 'faiss_index'.")