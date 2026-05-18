import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

# 1. Chargement dyal les clés mn fichier .env
load_dotenv()
endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

if not endpoint or not key:
    print("❌ L'Endpoint wla l'Key makayninش! تأكد من ملف .env")
    exit()

# 2. Initialisation dyal l'client Azure
client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# 3. Dossiers dyal l'input w l'output
data_dir = "data"
output_dir = "extracted_text"
os.makedirs(output_dir, exist_ok=True)

print("🚀 Bismillah! L'extraction bdat...\n")

# 4. Boucle 3la les PDFs kamlin f dossier 'data'
for filename in os.listdir(data_dir):
    if filename.endswith(".pdf"):
        file_path = os.path.join(data_dir, filename)
        print(f"⏳ Kan-traitiw: {filename} ...")
        
        with open(file_path, "rb") as f:
            # Nkhdmo b prebuilt-layout bach yjbed l'ktaba w les tableaux m9addin
            poller = client.begin_analyze_document(
                "prebuilt-layout", 
                body=f, 
                content_type="application/octet-stream"
            )
        
        result = poller.result()
        
        # 5. Sauvegarde dyal l'ktaba
        output_file = os.path.join(output_dir, filename.replace(".pdf", ".txt"))
        with open(output_file, "w", encoding="utf-8") as out_f:
            out_f.write(result.content)
            
        print(f"✅ Tssauvgarda f: {output_file}\n")

print("🎉 L'extraction salat b naja7 100%! L'data wajda l'RAG.")