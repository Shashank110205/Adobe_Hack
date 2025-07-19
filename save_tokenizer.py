from transformers import AutoTokenizer

model_id = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained("models/tokenizer")

print("✅ Tokenizer saved as tokenizer.json")
