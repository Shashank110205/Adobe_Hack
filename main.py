import os
import json
import numpy as np
import joblib
import re
import onnxruntime as ort
from tokenizers import Tokenizer
from unstructured.partition.pdf import partition_pdf
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

print("🔄 Loading model and classifier...")
session = ort.InferenceSession("models/all-MiniLM-L6-v2.onnx")
tokenizer = Tokenizer.from_file("models/tokenizer/tokenizer.json")
clf = joblib.load("models/heading_classifier.pkl")
le = joblib.load("models/label_encoder.pkl")

BLACKLIST_PHRASES = {
    "version 2014",
    "international software testing qualifications board",
    "page"
}

removed_entries = []
duplicate_entries = []
translation_cache = {}

def normalize(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u2013", "-").replace("\u00a9", "").strip()
    return text.lower()

def is_redundant(text):
    norm = normalize(text)
    if len(norm) < 3:
        return "too short"
    if any(bad in norm for bad in BLACKLIST_PHRASES):
        return f"blacklist: {norm}"
    if re.match(r"page \d+ of \d+", norm):
        return "page number"
    return None

def is_sentence_like(text):
    if not text or len(text.split()) < 3:
        return False
    return text[0].islower() or text.endswith(".")

# def translate_if_needed(text):
#     if text in translation_cache:
#         return translation_cache[text]
#     try:
#         lang = detect(text)
#         if lang != "en":
#             translated = GoogleTranslator(source='auto', target='en').translate(text)
#             translation_cache[text] = translated
#             return translated
#     except Exception as e:
#         print(f"🌐 Translation failed for '{text}': {e}")
#     translation_cache[text] = text
#     return text

def batch_embed(texts):
    encoded = [tokenizer.encode(t) for t in texts]
    input_ids = np.array([e.ids for e in encoded], dtype="int64")
    attention_mask = np.array([[1] * len(e.ids) for e in encoded], dtype="int64")
    outputs = session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })
    
    # Convert ONNX output to numpy array first, then slice
    output_array = np.array(outputs[0])
    return output_array[:, 0, :]  # CLS token

def get_levels(texts):
    levels = []
    embeddings = batch_embed(texts)
    for text, emb in zip(texts, embeddings):
        text = text.strip()
        match = re.match(r'^\d+(\.\d+)*\s+', text)
        if match:
            dot_count = text.split()[0].count(".") + 1
            if dot_count == 1:
                levels.append("H1")
                continue
            elif dot_count == 2:
                levels.append("H2")
                continue
            else:
                levels.append("H3")
                continue
        if len(text.split()) > 8 or text.endswith("."):
            levels.append("H2")
            continue
        try:
            pred = clf.predict([emb])
            levels.append(le.inverse_transform(pred)[0])
        except Exception as e:
            print(f"⚠️ ONNX error for '{text}': {e}")
            levels.append("H2")
    return levels

def extract_outline(pdf_path):
    print(f"📄 Processing: {pdf_path}")
    elements = partition_pdf(filename=pdf_path, infer_table_structure=False)
    title = os.path.basename(pdf_path).replace(".pdf", "")

    raw_texts = []
    page_nums = []

    for el in elements:
        if type(el).__name__ in ["Title", "SectionHeader"]:
            text = el.text.strip()
            if len(text) < 3:
                continue
            reason = is_redundant(text)
            if reason:
                removed_entries.append({"text": text, "reason": reason})
                continue
            if is_sentence_like(text):
                removed_entries.append({"text": text, "reason": "sentence-like"})
                continue
            key = normalize(text)
            if key in translation_cache:
                duplicate_entries.append(text)
                continue
            translation_cache[key] = None
            raw_texts.append(text)
            page_nums.append(getattr(el.metadata, "page_number", 1))

    levels = get_levels(raw_texts)
    outline = [{"level": lvl, "text": txt, "page": pg} for txt, pg, lvl in zip(raw_texts, page_nums, levels)]

    return {"title": title, "outline": outline}

def process_file(file_name):
    print(f"📄 Processing: {file_name}")
    input_path = os.path.join("/app/input", file_name)
    output_path = os.path.join("/app/output", file_name.replace(".pdf", ".json"))
    outline_data = extract_outline(input_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outline_data, f, indent=4, ensure_ascii=False)
    print(f"✅ Saved: {output_path}")

def main():
    start_time = time.time()
    
    # Ensure input and output directories exist as per challenge requirements
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    if not input_dir.exists():
        print("❌ Error: /app/input directory not found")
        return
    
    output_dir.mkdir(exist_ok=True)
    
    # Get all PDF files from input directory
    pdf_files = [f for f in os.listdir("/app/input") if f.endswith(".pdf")]
    
    if not pdf_files:
        print("⚠️ No PDF files found in /app/input")
        return
    
    print(f"📋 Found {len(pdf_files)} PDF files to process")
    
    # Process files using ThreadPoolExecutor for performance
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_file, pdf_files)
    
    # Optional logging (create logs directory if needed)
    logs_dir = Path("/app/logs")
    logs_dir.mkdir(exist_ok=True)
    
    with open("/app/logs/logs_removed.json", "w", encoding="utf-8") as f:
        json.dump(removed_entries, f, indent=2, ensure_ascii=False)
    with open("/app/logs/logs_duplicates.json", "w", encoding="utf-8") as f:
        json.dump(duplicate_entries, f, indent=2, ensure_ascii=False)
    
    end_time = time.time()
    print(f"🎯 Processing completed in {end_time - start_time:.2f} seconds")
    print(f"🧹 Removed: {len(removed_entries)} | 🔁 Duplicates skipped: {len(duplicate_entries)}")

if __name__ == "__main__":
    main()
