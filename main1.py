# this script extract pdf heading hardcoded simple but effective
# it uses unstructured library to extract headings from pdf files
# it uses PyPDF2 to extract metadata like title
# it uses sentence-transformers to classify headings into H1, H2, H3 levels
# it uses a simple regex to detect numbered headings 

import os
import json
import re
from unstructured.partition.pdf import partition_pdf
from PyPDF2 import PdfReader

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
    """
    Determines heading level based on:
    - Numbered format (e.g., 1., 1.2., 1.2.3)
    - Uppercase
    - Multilingual H1 keyword set
    - Default fallback: H2
    """
    # Normalize spacing and punctuation
    text_stripped = re.sub(r'\s+', ' ', text).strip()
    clean_text = text_stripped.rstrip(".:").lower()

    # 1. Numbered heading detection
    match = re.match(r'^(\d+(\.\d+){0,})\s+', text_stripped)
    if match:
        level_count = match.group(0).count('.') + 1
        if level_count == 1:
            return "H1"
        elif level_count == 2:
            return "H2"
        else:
            return "H3"

    # 2. All caps heading
    if text_stripped.isupper():
        return "H1"

    # 3. Multilingual top-level heading keywords
    h1_keywords = {
        # English
        "overview", "abstract", "features", "introduction", "conclusion",
        "summary", "background", "system design", "deployment",
        # Hindi
        "परिचय", "सारांश", "पृष्ठभूमि", "निष्कर्ष",
        # French
        "introduction", "résumé", "conclusion", "contexte",
        # Spanish
        "introducción", "resumen", "conclusión",
        # German
        "einleitung", "zusammenfassung", "schlussfolgerung"
    }

    if clean_text in h1_keywords:
        return "H1"

    # 4. Default fallback
    return "H2"



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

    # Optional: sort by page number and then text
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
