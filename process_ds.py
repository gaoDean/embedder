import os
import torch
import torch.nn.functional as F
import config as cfg
from datasets import load_dataset
from transformers import AutoTokenizer
from jina_inference import Jina

# 8 jina instances
num_proc = 0 # no multiple jina
num_proc_load_dataset = 8

def get_portions(paragraph, portion_length):

    portions = []

    for i in range(1, len(paragraph) // portion_length, 1):
        cur = paragraph[(i - 1) * portion_length : i * portion_length]
        splits = cur.split(" ")

        # everything except the first and last word since they might be cut off from the portioning
        processed = " ".join(splits[1:-1])

        portions.append(processed)

    return portions

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

    jina = Jina()

    def process(batch):
        """
        takes in a dataset batch

        returns {
            "input_ids": ...,
            "attention_mask": ...,
            "embeddings": ...,
        }
        """


        unprocessed_texts = batch["text"]

        portions_batch = []

        # the idea is that we split the paragraphs into "portions"
        # which are each one to two sentences long
        # so we can make better use of our dataset

        for text in unprocessed_texts:
            portions = get_portions(text, cfg.MAX_CHARS_TRUNC)
            portions_batch.append(portions)

            del portions

        tokenized = tokenizer(portions_batch, add_special_tokens=True, truncation=True, padding=cfg.TRUNC_LENGTH, max_length=cfg.TRUNC_LENGTH)

        with torch.no_grad():
            embeddings = F.layer_norm(jina.model(portions_batch, cfg.TRUNC_LENGTH), (cfg.CONTEXT_DIM,))
        tokenized["embeddings"] = embeddings.detach().cpu().numpy()

        return tokenized

    # tokenize the dataset
    tokenized = split_dataset.map(
            process,
            batched=True,
            batch_size=40,
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
