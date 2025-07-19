# This script extracts headings from PDF files and classifies them into H1, H2, H3 levels.
# It uses the unstructured library for PDF parsing, PyPDF2 for metadata extraction, 
# and a multilingual MiniLM model for heading classification.
# this uses direct model for inference and tokenizer loading.

import os
import json
import re
from unstructured.partition.pdf import partition_pdf
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util

# Load multilingual MiniLM model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Define multilingual anchor headings for each level
anchor_headings = {
    "H1": ["Introduction", "Overview", "परिचय", "Résumé", "Einleitung", "Deployment"],
    "H2": ["System Architecture", "Related Work", "Approach", "Background", "Architecture"],
    "H3": ["Implementation Details", "Model Layers", "Loss Function", "Data Preprocessing"]
}

# Precompute anchor embeddings
anchor_embeddings = {
    level: model.encode(texts, convert_to_tensor=True)
    for level, texts in anchor_headings.items()
}

def get_pdf_title(pdf_path):
    """Try to extract PDF title from metadata. Fallback to file name."""
    try:
        reader = PdfReader(pdf_path)
        if reader.metadata and reader.metadata.title:
            title = reader.metadata.title.strip()
            if title:
                return title
    except Exception:
        pass
    return os.path.basename(pdf_path).replace(".pdf", "")

def get_heading_level(text: str) -> str:
    """Use multilingual transformer to classify heading level."""
    if not text or len(text.strip()) < 2:
        return "H2"
    
    embedding = model.encode(text.strip(), convert_to_tensor=True)

    scores = {
        level: util.cos_sim(embedding, anchor_emb).max().item()
        for level, anchor_emb in anchor_embeddings.items()
    }

    best_level = max(scores, key=scores.get)
    return best_level

def extract_outline(pdf_path):
    print(f"🔍 Processing: {pdf_path}")
    elements = partition_pdf(filename=pdf_path, infer_table_structure=False)
    title = get_pdf_title(pdf_path)
    headings = []

    for el in elements:
        if hasattr(el, "category") and any(k in el.category for k in ["Header", "Title"]):
            text = el.text.strip()
            if len(text.strip()) == 0:
                continue
            page_num = getattr(el.metadata, "page_number", None) or 1
            level = get_heading_level(text)

            headings.append({
                "level": level,
                "text": text,
                "page": page_num
            })

    # Optional: sort by page number
    headings = sorted(headings, key=lambda x: x["page"])

    print(f"✅ Found {len(headings)} headings in '{title}'")
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
            with open(output_path, "w", encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
            print(f"📄 Saved: {output_path}\n")

if __name__ == "__main__":
    main()
