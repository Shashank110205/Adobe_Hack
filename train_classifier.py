import joblib
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load ONNX model + tokenizer
model_path = "models/minilm.onnx"
session = ort.InferenceSession(model_path)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Sample data — update with real examples if available
headings = [
    "Introduction",
    "System Architecture",
    "Approach",
    "Implementation Details",
    "परिचय",
    "Résumé",
    "Model Layers",
    "Background"
]
labels = ["H1", "H1", "H2", "H3", "H1", "H1", "H3", "H2"]  # Must match order

def encode(text):
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    input_ids = inputs["input_ids"].astype("int64")         # ✅ convert to int64
    attention_mask = inputs["attention_mask"].astype("int64")
    outputs = session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })
    return outputs[0][:, 0, :]  # CLS token

X = np.vstack([encode(h) for h in headings])
le = LabelEncoder()
y = le.fit_transform(labels)

# Train classifier
clf = LogisticRegression()
clf.fit(X, y)

# Save classifier and label encoder
joblib.dump(clf, "models/heading_classifier.pkl")
joblib.dump(le, "models/label_encoder.pkl") 
