import os
import re
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_alef_ar, normalize_teh_marbuta_ar, normalize_alef_maksura_ar

input_dir = "extracted_text"
output_dir = "cleaned_text"
os.makedirs(output_dir, exist_ok=True)

def clean_arabic_legal_text(text):
    # 1. T7yad l'7arakat (Dediacritization)
    text = dediac_ar(text)
    
    # 2. Normalisation dyal l'7ourouf l'3arbiya
    text = normalize_alef_ar(text)          # (أ, إ, آ) -> ا
    text = normalize_teh_marbuta_ar(text)   # ة -> ه (Aw l'3eks 3la 7sab l'besoin, hna nrodouha standard)
    text = normalize_alef_maksura_ar(text)  # ى -> ي
    
    # 3. T7yad l'espaces zaydin w l-khrbiq
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

print("🧹 Bdat 3amaliyat l'Nettoyage b camel-tools...\n")

for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
            raw_text = f.read()
            
        cleaned = clean_arabic_legal_text(raw_text)
        
        output_file = os.path.join(output_dir, filename)
        with open(output_file, "w", encoding="utf-8") as out_f:
            out_f.write(cleaned)
            
        print(f"✅ Tna9a w t-sauvgarda: {filename}")

print("\n✨ L'data daba n9iya 100% w wajda l'Chunking!")