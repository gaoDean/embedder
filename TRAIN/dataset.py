from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from jina_inference import Jina
import config
import os
import re

class HFDataset(Dataset):
    def __init__(self, llm_tokenizer, split="train"):
        self.llm_tokenizer = llm_tokenizer

        if os.path.exists(config.DATASET_CACHE_DIR):
            print(f"Loading cached dataset from {config.DATASET_CACHE_DIR}...")
            self.ds = load_from_disk(config.DATASET_CACHE_DIR)
        else:
            print("Processing dataset (no cache found) ...")
            ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=split)
            jina = Jina()

            def dataset_map_fn(batch):
                """
                takes in a dataset batch

                returns {
                    "input_ids": ...,
                    "attention_mask": ...,
                    "embeddings": ...,
                }
                """


                texts = batch["article"]

                for i, entry in enumerate(texts):
                    # e.g NEW YORK (CNN) -- A massive anti-Ma...
                    res = re.findall(r"^.*\s--\s", entry)

                    # remove the news handle metadata
                    if res != []:
                        texts[i] = texts[i][len(res[0]) :]

                    # limit article length to config value
                    texts[i] = texts[i][:config.DATASET_ENTRY_LENGTH_LETTERS]

                    # remove any unfinished words due to length trimming
                    texts[i] = re.sub(r"\s\S*$", "", texts[i])

                tokenized = self.llm_tokenizer(texts, add_special_tokens=True, truncation=False)
                tokenized["embeddings"] = jina.embed(texts)

                return tokenized

            self.ds = ds.map(
                dataset_map_fn,
                batched = True,
                num_proc = 1, # because we're doing inference
                remove_columns=ds.column_names
            )

            print(f"Saving dataset to {config.DATASET_CACHE_DIR} ...")
            self.ds.save_to_disk(config.DATASET_CACHE_DIR)

    def __getitem__(self, i):
        '''
        returns (array) ref tokens, (array) target tokens for a given i
        '''
        entry = self.ds[i]

        tokens = entry["input_ids"]

        ref_toks = tokens[:-1] # everythign except very last
        target_toks = tokens[1:] # everything except very first

        return torch.tensor(ref_toks), torch.tensor(target_toks)

    def __len__(self):
        return len(self.ds)

def collate_fn(batch, pad_id=0):
    ref_toks_batch, target_toks_batch = zip(*batch)
    max_len = max(len(entry) for entry in ref_toks_batch)
    padded_x = torch.full(
        (len(ref_toks_batch), max_len),
        pad_id,
        dtype=torch.long
    )
    padded_y = torch.full(
        (len(target_toks_batch), max_len),
        pad_id,
        dtype=torch.long
    )

    for i, (x, y) in enumerate(zip(ref_toks_batch, target_toks_batch)):
        padded_x[i, :len(x)] = x
        padded_y[i, :len(y)] = y
    return padded_x, padded_y

def get_dataloader(tokenizer, split, shuffle=True):
    dataset = HFDataset(tokenizer, split)
    return DataLoader(
        dataset, batch_size=config.DATALOADER_BATCHSIZE, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
