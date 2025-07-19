# This script saves the tokenizer for the ONNX model used in the heading extraction pipeline.
# It uses the AutoTokenizer from the transformers library to load the tokenizer
# and saves it in the specified directory.

from transformers import AutoTokenizer

model_id = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained("models/tokenizer")

print("✅ Tokenizer saved as tokenizer.json")
