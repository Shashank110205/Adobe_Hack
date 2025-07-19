# This script exports a HuggingFace model to ONNX format.
# It uses the transformers library to load the model and tokenizer,
# and torch to export the model to ONNX.
# The exported model can be used for inference in various environments that support ONNX.
# It exports the "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" model.

from transformers import AutoTokenizer, AutoModel
import torch
import os

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# Create dummy input
text = "Introduction"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Make sure output folder exists
os.makedirs("models", exist_ok=True)

# Export to ONNX
torch.onnx.export(
    model,
    (input_ids, attention_mask),
    "models/minilm.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "last_hidden_state": {0: "batch", 1: "seq"},
    },
    opset_version=17
)

print("✅ Exported to models/minilm.onnx")
