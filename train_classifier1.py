import json
import os
import joblib
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "models/all-MiniLM-L6-v2.onnx"
TOKENIZER_PATH = "models/tokenizer/tokenizer.json"
TRAIN_DATA = "data/heading_training_data.json"

print("🔄 Loading tokenizer and ONNX model...")
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
session = ort.InferenceSession(MODEL_PATH)

def embed(text):
    output = tokenizer.encode(text)
    input_ids = np.array([output.ids], dtype="int64")
    attention_mask = np.array([[1] * len(output.ids)], dtype="int64")
    outputs = session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })
    return outputs[0][:, 0, :]  # CLS token

with open(TRAIN_DATA, "r", encoding="utf-8") as f:
    samples = json.load(f)

samples = [s for s in samples if s["text"].strip() and s["label"] in {"H1", "H2", "H3"}]
texts = [s["text"] for s in samples]
labels = [s["label"] for s in samples]

print("🧠 Embedding headings...")
X = np.vstack([embed(t) for t in texts])
le = LabelEncoder()
y = le.fit_transform(labels)

print("🎯 Training classifier...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/heading_classifier.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("✅ Classifier and label encoder saved!")
