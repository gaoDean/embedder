import os
import torch
import torch.nn.functional as F
import config as cfg
from datasets import load_dataset
from transformers import AutoTokenizer
from jina_inference import Jina

# 8 jina instances
num_proc = 8 # no multiple jina
num_proc_load_dataset = 8

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
    e_tokenizer = AutoTokenizer.from_pretrained(cfg.EMBEDDING_MODEL_NAME, trust_remote_code=True)

    if os.path.exists(cfg.DATASET_CACHE_DIR):
        print("dataset already exists")
        return None

    def process(batch):
        """
        takes in a dataset batch

        returns {
            "input_ids": ...,
            "attention_mask": ...,
            "embeddings": ...,
        }
        """


        texts = batch["text"]

        for i, entry in enumerate(texts):
            if len(entry) > cfg.MAX_TEXT_LENGTH:
                texts[i] = texts[i][:cfg.MAX_TEXT_LENGTH]

        tokenized = tokenizer(texts, add_special_tokens=True, truncation=False)
        e_tokenized = e_tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=True,
            return_tensors="pt"
        )

        tokenized["e_input_ids"] = e_tokenized["input_ids"]
        tokenized["e_attention_mask"] = e_tokenized["attention_mask"]

        return tokenized

    # tokenize the dataset
    tokenized = split_dataset.map(
            process,
            batched=True,
            remove_columns=['text'],
            desc="processing dataset",
            num_proc=num_proc,
            )

    tokenized.save_to_disk(cfg.DATASET_CACHE_DIR)

    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)

if __name__ == '__main__':
    main()
