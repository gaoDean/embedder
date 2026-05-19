import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import JinaEvaluator

def build_thesaurus(model):
    """Pre-computes similarity between vocabulary tokens using model's input embeddings."""
    embeddings = model.get_input_embeddings().weight.detach() # [VocabSize, Hidden]
    # Normalize for cosine similarity
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def precompute_dataset(args):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Pythia Model & Tokenizer...")
    pythia_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    pythia_tokenizer.pad_token = pythia_tokenizer.eos_token
    pythia_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped").to(device)

    print("Loading Jina Tokenizer...")
    jina_eval = JinaEvaluator(device) # We just need the tokenizer from it

    # 1. Precompute Thesaurus
    print("Building and saving thesaurus...")
    thesaurus = build_thesaurus(pythia_model)
    torch.save(thesaurus.cpu(), os.path.join(args.output_dir, "thesaurus.pt"))

    # 2. Pre-tokenize Dataset
    # NOTE: In your train.py, you use simulated random embeddings (torch.randn).
    # Since there's no actual text dataset loaded in train.py yet, here is the 
    # scaffolding for when you connect your real text dataset.
    
    # Example dataset structure (Replace with your actual data loading, e.g., datasets.load_dataset)
    dummy_text_data = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Another example text to process."
    ]

    print("Pre-tokenizing dataset...")
    processed_data = []

    for text in tqdm(dummy_text_data):
        # Pythia tokenization (No special tokens, get offsets)
        p_enc = pythia_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        
        # Jina tokenization (With special tokens, get offsets)
        j_enc = jina_eval.tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)

        processed_data.append({
            "text": text,
            "pythia_input_ids": p_enc['input_ids'],
            "pythia_offsets": p_enc['offset_mapping'],
            "jina_input_ids": j_enc['input_ids'],
            "jina_offsets": j_enc['offset_mapping']
        })

    # Save processed dataset
    torch.save(processed_data, os.path.join(args.output_dir, "tokenized_dataset.pt"))
    print(f"Done! Saved precomputed data to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./precomputed_data", help="Directory to save precomputed tensors")
    args = parser.parse_args()
    precompute_dataset(args)
