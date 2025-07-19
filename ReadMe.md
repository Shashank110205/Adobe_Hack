# 🧾 PDF Heading Extractor (Multilingual & Optimized)

This project extracts structured outlines from PDF documents by detecting headings and classifying them into hierarchical levels (H1, H2, H3). The solution is designed to be efficient, multilingual-aware, and ready for scaling across diverse document types.

---

## 🔍 What This Project Does

This tool:

* Scans PDFs for structural headings like titles and section headers
* Detects the language of each heading
* Translates non-English headings into English
* Embeds the translated headings using a fast ONNX model
* Predicts their structure level using a trained machine learning classifier
* Outputs a clean, structured JSON outline for each PDF

---

## ✅ Key Features

| Feature                    | Description                                                                   |
| -------------------------- | ----------------------------------------------------------------------------- |
| 🌍 Multilingual Support    | Automatically detects and translates non-English headings                     |
| ⚡ Optimized Inference      | Uses ONNX and batch processing for faster embedding                           |
| 🧠 Accurate Classification | Predicts heading levels (H1, H2, H3) with a trained Logistic Regression model |
| 🧹 Noise Removal           | Filters out non-structural text (e.g., page numbers, footers)                 |
| 🔁 Duplicate Detection     | Skips repeated headings to avoid clutter                                      |
| 🧾 Logging                 | Keeps logs of all removed and duplicate entries                               |
| 🚀 Scalable                | Supports concurrent processing of multiple PDFs                               |

---

## 📁 Project Structure

```
📦 Project Root
├── 📂 input/                # Drop your PDFs here
├── 📂 output/               # JSON output with cleaned heading outlines
├── 📂 models/               # Contains ONNX model and classifier artifacts
│   ├── all-MiniLM-L6-v2.onnx
│   ├── tokenizer/tokenizer.json
│   ├── heading_classifier.pkl
│   └── label_encoder.pkl
├── 🧠 main.py              # Main script to process PDFs
├── 📄 logs_removed.json     # Headings removed with reasons
└── 📄 logs_duplicates.json  # Duplicates skipped during processing
```

---

## 🧠 How It Works

1. **PDF Parsing**

   * Uses `unstructured` to extract text elements from each PDF

2. **Filtering**

   * Filters out footers, very short lines, and irrelevant sections

3. **Language Handling**

   * Detects the language of each heading
   * Translates non-English text into English using `deep-translator`

4. **Embedding**

   * Uses the ONNX version of `all-MiniLM-L6-v2` to embed headings
   * Batch processes headings for performance

5. **Prediction**

   * Feeds embeddings to a trained `LogisticRegression` model
   * Classifies each heading as H1, H2, or H3

6. **Output**

   * Saves structured JSON with headings, page numbers, and levels

---

## 🧪 Example Output

```json
{
  "title": "sample",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "1.1 What is AI?", "page": 2 },
    { "level": "H3", "text": "History of AI", "page": 3 }
  ]
}
```

---

## 🛠 Technologies Used

* **ONNX Runtime** — Fast transformer inference engine
* **HuggingFace Tokenizers** — Tokenizing text for ONNX input
* **scikit-learn** — Heading level classifier
* **langdetect** — Detects language of headings
* **deep-translator** — Translates headings to English
* **unstructured** — Parses PDFs into structured elements
* **concurrent.futures** — Enables multi-threaded PDF processing

---

## 🧩 Future Enhancements

* [ ] Include both original and translated headings in output JSON
* [ ] Add `"lang"` field to each heading for metadata
* [ ] Deploy as a web app using FastAPI or Streamlit
* [ ] Package as a command-line tool for wider usage

---

## 📌 Why This Project is Valuable

This tool is ideal for anyone needing:

* Automatic table-of-contents extraction from academic or technical PDFs
* Support for multilingual documents (e.g., Hindi, Arabic, Chinese)
* Clean, structured outlines for indexing, summarization, or navigation

---

## 🙌 Summary

You’ve built a multilingual, ONNX-accelerated, production-ready PDF heading extractor that’s:

* ⚡ Fast
* 🌍 Language-aware
* 🧠 Smart
* 🧹 Clean and reliable

> Ready to be deployed or extended into real-world products.
