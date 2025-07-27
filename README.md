# 🧾 PDF Heading Extractor

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
   Uses `unstructured` to extract text elements from each PDF.

2. **Filtering**
   Filters out footers, very short lines, and irrelevant sections.

3. **Language Handling**
   Detects the language of each heading and translates to English using `deep-translator`.

4. **Embedding**
   Uses ONNX version of `all-MiniLM-L6-v2` for fast vector embedding.

5. **Prediction**
   Classifies headings into H1/H2/H3 via trained `LogisticRegression` model.

6. **Output**
   Produces structured JSON with heading text, page number, and level.

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

## ⚙️ Installation & Setup

### 🔧 Prerequisites

Make sure you have Docker and Docker Compose installed.

### 🚀 Build & Run

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/pdf-heading-extractor
   cd pdf-heading-extractor
   ```

2. **Build the Docker image:**

   ```bash
   docker compose build
   ```

3. **Start the service:**

   ```bash
   docker compose up
   ```

> 📴 Once the image is built, the application can be run completely **offline** — no internet is required to extract headings from PDFs.

### 🔗 Accessing the App

After the container is running, the API will be available at:

```
http://localhost:8000/
```

>[!TIP]
> You can interact with the API using the Swagger UI. The UI is available at:
>
>```
>http://localhost:8000/docs
>```

---

## 🧩 Future Enhancements

* [ ] Include both original and translated headings in output JSON
* [ ] Add `"lang"` field to each heading for metadata
* [ ] Deploy as a web app using FastAPI or Streamlit
* [ ] Package as a command-line tool for wider usage

---

## 📌 Why This Project is Valuable

This tool is ideal for:

* Automatic table-of-contents extraction from academic or technical PDFs
* Supporting multilingual documents (e.g., Hindi, Arabic, Chinese)
* Creating clean, structured outlines for indexing, summarization, or navigation

---

## 🙌 Summary

You’ve built a multilingual, ONNX-accelerated, production-ready PDF heading extractor that’s:

* ⚡ Fast
* 🌍 Language-aware
* 🧠 Smart
* 🧹 Clean and reliable

Ready to be deployed or extended into real-world products.

---

Let me know if you'd like to include usage examples (e.g. `curl` or Python client) or a FastAPI-based UI section!

