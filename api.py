import os
import numpy as np
import joblib
import re
import onnxruntime as ort
from tokenizers import Tokenizer
from unstructured.partition.pdf import partition_pdf
from deep_translator import GoogleTranslator
from langdetect import detect
import tempfile
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="PDF Heading Extractor API",
    description="Extract and classify headings from PDF files into H1, H2, H3 levels",
    version="1.0.0"
)

# Global variables for models (loaded once at startup)
session = None
tokenizer = None
clf = None
le = None

BLACKLIST_PHRASES = {
    "version 2014",
    "international software testing qualifications board",
    "page"
}

# Global lists for logging (you might want to use a proper logging system in production)
removed_entries = []
duplicate_entries = []
translation_cache = {}

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global session, tokenizer, clf, le
    print("🔄 Loading model and classifier...")
    try:
        session = ort.InferenceSession("models/all-MiniLM-L6-v2.onnx")
        tokenizer = Tokenizer.from_file("models/tokenizer/tokenizer.json")
        clf = joblib.load("models/heading_classifier.pkl")
        le = joblib.load("models/label_encoder.pkl")
        print("✅ Models loaded successfully!")
        
        # Verify models are actually loaded
        if tokenizer is None:
            raise Exception("Tokenizer failed to load")
        if session is None:
            raise Exception("ONNX session failed to load")
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        # Don't raise here - let the server start but handle in endpoints
        print("⚠️ Server will start but models are not loaded")

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

def translate_if_needed(text):
    if text in translation_cache:
        return translation_cache[text]
    try:
        lang = detect(text)
        if lang != "en":
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            translation_cache[text] = translated
            return translated
    except Exception as e:
        print(f"🌐 Translation failed for '{text}': {e}")
    translation_cache[text] = text
    return text

def batch_embed(texts):
    global tokenizer, session
    
    # Check if models are loaded
    if tokenizer is None:
        raise HTTPException(status_code=500, detail="Tokenizer not loaded. Server may still be starting up.")
    if session is None:
        raise HTTPException(status_code=500, detail="ONNX session not loaded. Server may still be starting up.")
    
    encoded = [tokenizer.encode(translate_if_needed(t)) for t in texts]
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
    global clf, le
    
    # Check if classifier and label encoder are loaded
    if clf is None:
        raise HTTPException(status_code=500, detail="Classifier not loaded. Server may still be starting up.")
    if le is None:
        raise HTTPException(status_code=500, detail="Label encoder not loaded. Server may still be starting up.")
    
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

def extract_outline_from_file(file_content: bytes, filename: str):
    """Extract outline from PDF file content"""
    print(f"📄 Processing: {filename}")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_content)
        temp_path = temp_file.name
    
    try:
        elements = partition_pdf(filename=temp_path, infer_table_structure=False)
        title = filename.replace(".pdf", "")

        raw_texts = []
        page_nums = []
        current_removed = []
        current_duplicates = []
        current_cache = {}

        for el in elements:
            if type(el).__name__ in ["Title", "SectionHeader"]:
                text = el.text.strip()
                if len(text) < 3:
                    continue
                reason = is_redundant(text)
                if reason:
                    current_removed.append({"text": text, "reason": reason})
                    continue
                if is_sentence_like(text):
                    current_removed.append({"text": text, "reason": "sentence-like"})
                    continue
                key = normalize(text)
                if key in current_cache:
                    current_duplicates.append(text)
                    continue
                current_cache[key] = None
                raw_texts.append(text)
                page_nums.append(getattr(el.metadata, "page_number", 1))

        levels = get_levels(raw_texts)
        outline = [{"level": lvl, "text": txt, "page": pg} for txt, pg, lvl in zip(raw_texts, page_nums, levels)]

        result = {
            "title": title,
            "outline": outline,
            "processing_info": {
                "total_headings_found": len(outline),
                "removed_entries": len(current_removed),
                "duplicate_entries": len(current_duplicates)
            }
        }
        
        return result
    
    finally:
        # Clean up temporary file
        os.unlink(temp_path)

@app.post("/extract-headings")
async def extract_headings(file: UploadFile = File(...)):
    # Check if models are loaded
    if tokenizer is None or session is None or clf is None or le is None:
        raise HTTPException(
            status_code=503, 
            detail="Server is still loading models. Please wait a moment and try again."
        )
    
    # Get filename with fallback
    filename: str = file.filename or "uploaded_file.pdf"
    
    # Validate file type
    if not filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Process the PDF
        result = extract_outline_from_file(file_content, filename)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        print(f"❌ Error processing file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/extract-headings-batch")
async def extract_headings_batch(files: List[UploadFile] = File(...)):
    """
    Extract headings from multiple PDF files.
    
    - **files**: List of PDF files to process
    
    Returns JSON with extracted headings for each file.
    """
    # Check if models are loaded
    if tokenizer is None or session is None or clf is None or le is None:
        raise HTTPException(
            status_code=503, 
            detail="Server is still loading models. Please wait a moment and try again."
        )
    
    results = []
    
    for i, file in enumerate(files):
        # Handle missing filename
        if not file.filename:
            results.append({
                "filename": f"file_{i+1}.pdf",
                "error": "No filename provided"
            })
            continue
        
        filename: str = file.filename
        
        if not filename.endswith('.pdf'):
            results.append({
                "filename": filename,
                "error": "File must be a PDF"
            })
            continue
        
        try:
            file_content = await file.read()
            result = extract_outline_from_file(file_content, filename)
            result["filename"] = filename
            results.append(result)
        
        except Exception as e:
            print(f"❌ Error processing file {filename}: {e}")
            results.append({
                "filename": filename,
                "error": f"Error processing PDF: {str(e)}"
            })
    
    return JSONResponse(content={"results": results})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": session is not None}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PDF Heading Extractor API",
        "version": "1.0.0",
        "endpoints": {
            "/extract-headings": "POST - Extract headings from a single PDF",
            "/extract-headings-batch": "POST - Extract headings from multiple PDFs",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "api:app",  # Replace "main" with your actual filename
        host="0.0.0.0",
        port=8000,
        reload=True
    )
