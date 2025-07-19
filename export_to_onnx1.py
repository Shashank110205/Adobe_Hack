import torch
from transformers import AutoModel, AutoTokenizer
import os

# Model ID from HuggingFace
model_id = "sentence-transformers/all-MiniLM-L6-v2"

# Load model and set to evaluation mode
model = AutoModel.from_pretrained(model_id)
model.eval()

# Load tokenizer and prepare input tensors
tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer(
    "Hello world", 
    return_tensors="pt", 
    padding=True, 
    truncation=True
)

# Create directory for ONNX model
os.makedirs("models", exist_ok=True)

# Export the model to ONNX format
torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    "models/all-MiniLM-L6-v2.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state", "pooler_output"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "last_hidden_state": {0: "batch", 1: "seq"},
        "pooler_output": {0: "batch"}
    },
    opset_version=17
)

print("✅ ONNX export complete!")
