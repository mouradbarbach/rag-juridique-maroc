import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

input_dir = "cleaned_text"
output_file = "chunks.json"

# Configuration dyal l'Chunker
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

all_chunks = []
total_chunks = 0

print("🔪 L'Chunking bda...\n")

for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
            text = f.read()
        
        # T9ti3 dyal l'text
        chunks = text_splitter.split_text(text)
        total_chunks += len(chunks)
        
        # Sauvegarde m3a l'Metadata (Smiyat l'fichier w r-rqm dyal l'morceau)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": filename.replace(".txt", ".pdf"),
                "chunk_id": i,
                "text": chunk
            })
        
        print(f"✅ {filename}: T9sem l {len(chunks)} morceaux.")

# Sauvegarde f JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=4)

print(f"\n🎉 Salina! {total_chunks} chunks t-sauvegardaw mzyan f {output_file}.")
print("L'data daba wajda 100% bach t-dkhl l'FAISS Vector Database!")