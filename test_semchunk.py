import semchunk
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
chunker = semchunk.chunkerify(tokenizer, 10)

texts = ["Hello world, this is a test.", "Another test string for testing."]

try:
    chunks = chunker(texts)
    print("Chunker output:", chunks)
except Exception as e:
    print("Error:", e)
