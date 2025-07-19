# This script extracts headings from PDF files and classifies them into H1, H2, H3 levels.
# It uses the unstructured library for PDF parsing, PyPDF2 for metadata extraction,
# and a multilingual MiniLM model for heading classification.
# This version uses ONNX model inference and tokenizer loading. 
# It also includes a cleaning step to remove redundant entries and deduplicate headings.
# It supports multilingual headings and uses a simple regex for numbered headings.
# It also includes translation for non-English headings and logs removed and duplicate entries.



import os
import json
import numpy as np
import joblib
import re
import onnxruntime as ort
from tokenizers import Tokenizer
from unstructured.partition.pdf import partition_pdf
from deep_translator import GoogleTranslator
from langdetect import detect

print("🔄 Loading model and classifier...")
session = ort.InferenceSession("models/all-MiniLM-L6-v2.onnx")
tokenizer = Tokenizer.from_file("models/tokenizer/tokenizer.json")
clf = joblib.load("models/heading_classifier.pkl")
le = joblib.load("models/label_encoder.pkl")

# Blacklist text fragments
BLACKLIST_PHRASES = {
    "version 2014",
    "international software testing qualifications board",
    "page"
}

# Logging
removed_entries = []
duplicate_entries = []

def normalize(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("–", "-").replace("©", "").strip()
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

def clean_outline(data):
    seen = set()
    cleaned = []

    for item in data.get("outline", []):
        text = item.get("text", "").strip()
        reason = is_redundant(text)
        if reason:
            removed_entries.append({"text": text, "reason": reason})
            continue
        if is_sentence_like(text):
            removed_entries.append({"text": text, "reason": "sentence-like"})
            continue
        key = normalize(text)
        if key in seen:
            duplicate_entries.append(text)
            continue
        seen.add(key)
        cleaned.append(item)

    data["outline"] = cleaned
    return data

def translate_if_needed(text):
    try:
        lang = detect(text)
        if lang != "en":
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            return translated
    except Exception as e:
        print(f"🌐 Translation failed for '{text}': {e}")
    return text

def embed(text):
    translated = translate_if_needed(text)
    output = tokenizer.encode(translated)
    input_ids = np.array([output.ids], dtype="int64")
    attention_mask = np.array([[1] * len(output.ids)], dtype="int64")
    outputs = session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })
    return outputs[0][:, 0, :]  # CLS token

def get_heading_level(text):
    text = text.strip()
    match = re.match(r'^(\d+(.\d+){0,})\s+', text)
    if match:
        dot_count = text.split()[0].count(".") + 1
        if dot_count == 1:
            return "H1"
        elif dot_count == 2:
            return "H2"
        else:
            return "H3"
    if len(text.split()) > 8 or text.endswith("."):
        return "H2"
    try:
        pred = clf.predict(embed(text))
        return le.inverse_transform(pred)[0]
    except Exception as e:
        print(f"⚠️ ONNX error for '{text}': {e}")
        return "H2"

def extract_outline(pdf_path):
    print(f"📄 Processing: {pdf_path}")
    elements = partition_pdf(filename=pdf_path, infer_table_structure=False)
    headings = []
    title = os.path.basename(pdf_path).replace(".pdf", "")

    for el in elements:
        if type(el).__name__ in ["Title", "SectionHeader"]:
            text = el.text.strip()
            if len(text) < 3:
                continue
            page_num = getattr(el.metadata, "page_number", None) or 1
            level = get_heading_level(text)
            headings.append({
                "level": level,
                "text": text,
                "page": page_num
            })

    return {
        "title": title,
        "outline": headings
    }

def main():
    input_folder = "input"
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".pdf"):
            input_path = os.path.join(input_folder, file_name)
            output_data = extract_outline(input_path)

            # 🔍 Clean & deduplicate
            output_data = clean_outline(output_data)

            output_path = os.path.join(output_folder, file_name.replace(".pdf", ".json"))
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)

            print(f"✅ Saved cleaned: {output_path}")

    # Optional: Log removed & duplicate entries
    with open("logs/logs_removed.json", "w", encoding="utf-8") as f:
        json.dump(removed_entries, f, indent=2, ensure_ascii=False)
    with open("logs/logs_duplicates.json", "w", encoding="utf-8") as f:
        json.dump(duplicate_entries, f, indent=2, ensure_ascii=False)
    print(f"🧹 Removed entries: {len(removed_entries)} | 🔁 Duplicates skipped: {len(duplicate_entries)}")

if __name__ == "__main__":
    main()


