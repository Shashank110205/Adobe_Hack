# This script extracts headings from PDF files and classifies them into H1, H2, H3 levels.
# It uses the unstructured library for PDF parsing, PyPDF2 for metadata extraction, 
# and a multilingual MiniLM model for heading classification.
# this is a modified version of the original script to include ONNX model inference and tokenizer loading.


import os
import json
import numpy as np
import joblib
import re
import onnxruntime as ort
from transformers import AutoTokenizer
from unstructured.partition.pdf import partition_pdf

# Load tokenizer, ONNX model and classifier
print("🔄 Loading model and classifier...")
session = ort.InferenceSession("models/minilm.onnx")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
clf = joblib.load("models/heading_classifier.pkl")
le = joblib.load("models/label_encoder.pkl")

def get_heading_level(text: str) -> str:
    text = text.strip()

    # Rule-based: numbered heading override
    match = re.match(r'^(\d+(\.\d+){0,})\s+', text)
    if match:
        level_count = text.split()[0].count('.') + 1
        if level_count == 1:
            return "H1"
        elif level_count == 2:
            return "H2"
        else:
            return "H3"

    # Heuristic: long sentence likely not a heading
    if len(text.split()) > 8 or text.endswith("."):
        return "H2"

    try:
        # ML-based fallback
        inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
        input_ids = inputs["input_ids"].astype("int64")
        attention_mask = inputs["attention_mask"].astype("int64")
        outputs = session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
        embedding = outputs[0][:, 0, :]
        pred = clf.predict(embedding)
        return le.inverse_transform(pred)[0]
    except Exception as e:
        print(f"⚠️ ONNX failed on '{text}': {e}")
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

            output_path = os.path.join(output_folder, file_name.replace(".pdf", ".json"))
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)

            print(f"✅ Saved: {output_path}")

if __name__ == "__main__":
    main()
