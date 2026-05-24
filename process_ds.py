import os
import torch
import torch.nn.functional as F
import config as cfg
from datasets import load_dataset
from transformers import AutoTokenizer
from jina_inference import Jina

# 4 jina instances for parallel embedding extraction on GPU
num_proc = 4
num_proc_load_dataset = 8

# Global variable for lazy initialization inside worker processes
jina_instance = None

def tokenize_only(batch, tokenizer=None, max_text_length=None):
    texts = batch["text"]

    for i, entry in enumerate(texts):
        if len(entry) > max_text_length:
            texts[i] = texts[i][:max_text_length]

    tokenized = tokenizer(texts, add_special_tokens=True, truncation=False)
    tokenized["text"] = texts
    return tokenized

def embed_only(batch):
    global jina_instance
    if jina_instance is None:
        jina_instance = Jina()
    texts = batch["text"]
    with torch.inference_mode():
        embeddings = F.layer_norm(jina_instance.model(texts), (cfg.CONTEXT_DIM,))
    return {"embeddings": embeddings.cpu().numpy()}

def main():
    dataset = load_dataset("Skylion007/openwebtext", num_proc=num_proc_load_dataset)
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)

    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     test: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })

    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)

    if os.path.exists(cfg.DATASET_CACHE_DIR):
        print("dataset already exists")
        return None

    # 1. Parallel CPU-bound tokenization
    tokenized = split_dataset.map(
        tokenize_only,
        batched=True,
        batch_size=1000,
        desc="Tokenizing dataset",
        num_proc=num_proc_load_dataset,
        fn_kwargs={"tokenizer": tokenizer, "max_text_length": cfg.MAX_TEXT_LENGTH}
    )

    # 2. Parallel GPU-bound embedding extraction (using lazy Jina instantiation in workers)
    tokenized = tokenized.map(
        embed_only,
        batched=True,
        batch_size=256,
        remove_columns=['text'],
        desc="Generating embeddings",
        num_proc=num_proc,
    )

    tokenized.save_to_disk(cfg.DATASET_CACHE_DIR)

    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)

if __name__ == '__main__':
    main()
