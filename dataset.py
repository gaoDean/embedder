import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch.nn.functional as F
from jina_inference import Jina
import config as cfg
import os
import re

def clean_ds_entry(entry):
    # e.g NEW YORK (CNN) -- A massive anti-Ma...
    res = re.findall(r"^.*\s--\s", entry)

    # remove the news handle metadata
    if res != []:
        entry = entry[len(res[0]) :]

    # limit article length to config value
    entry = entry[:cfg.DATASET_ENTRY_LENGTH_LETTERS]

    # remove any unfinished words due to length trimming
    entry = re.sub(r"\s\S*$", "", entry)

    return entry

class HFDataset(Dataset):
    def __init__(self, llm_tokenizer, split="train"):
        self.llm_tokenizer = llm_tokenizer

        # Sanitize split name to keep caching distinct for each split
        sanitized_split = re.sub(r'[^a-zA-Z0-9]', '_', split)
        cache_dir = f"{cfg.DATASET_CACHE_DIR}_{sanitized_split}"

        if os.path.exists(cache_dir):
            print(f"Loading cached dataset from {cache_dir}...")
            self.ds = load_from_disk(cache_dir)
        else:
            print("Processing dataset (no cache found) ...")
            ds = load_dataset("openwebtext", , split=split)
            jina = Jina()

            # print(ds[0])
            # {'article': '...', 'highlights': '...', 'id': ...}

            # TESTING START
            # print(jina.embed([clean_ds_entry(ds[0]['article'])]))
            # print(self.llm_tokenizer([clean_ds_entry(ds[0]['article'])]))
            # TESTING END

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
                    texts[i] = clean_ds_entry(entry)

                tokenized = self.llm_tokenizer(texts, add_special_tokens=True, truncation=False)

                embeddings = F.layer_norm(jina.embed(texts), (cfg.CONTEXT_DIM,))
                tokenized["embeddings"] = embeddings

                return tokenized

            self.ds = ds.map(
                dataset_map_fn,
                batched = True,
                load_from_cache_file=False, # llm call cant be optimised
                num_proc = None, # because we're doing inference
                remove_columns=ds.column_names
            )

            print(f"Saving dataset to {cache_dir} ...")
            self.ds.save_to_disk(cache_dir)

    def __getitem__(self, i):
        '''
        returns (array) ref tokens, (array) target tokens for a given i
        '''
        entry = self.ds[i]

        tokens = entry["input_ids"]
        attn_mask = entry["attention_mask"]
        embedding = entry["embeddings"]

        ref_toks = tokens[:-1] # everythign except very last
        ref_mask = attn_mask[:-1]
        target_toks = tokens[1:] # everything except very first

        return (
            torch.tensor(ref_toks),
            torch.tensor(target_toks),
            torch.tensor(ref_mask),
            torch.tensor(embedding)
        )

    def __len__(self):
        return len(self.ds)

def collate_fn(batch, pad_id=0):
    ref_toks_batch, target_toks_batch, mask_batch, embedding_batch = zip(*batch)
    max_len = max(len(entry) for entry in ref_toks_batch)

    # Round max_len up to the nearest multiple of 8 to reduce the number of unique tensor shapes.
    # This prevents the MPS backend on Apple Silicon from constantly compiling new execution graphs.
    max_len = ((max_len + 7) // 8) * 8

    padded_x = torch.full(
        (len(ref_toks_batch), max_len),
        pad_id,
        dtype=torch.long
    )
    padded_y = torch.full(
        (len(target_toks_batch), max_len),
        -100, # ignore padding tokens when calculating cross entropy loss
        dtype=torch.long
    )
    padded_mask = torch.zeros(
        (len(mask_batch), max_len),
        dtype=torch.long
    )

    # Stack the embeddings to preserve individual context vectors for each batch item
    embedding_processed = embedding_batch[0].repeat(len(mask_batch), 1)

    for i, (x, y, m) in enumerate(zip(ref_toks_batch, target_toks_batch, mask_batch)):
        padded_x[i, :len(x)] = x
        padded_y[i, :len(y)] = y
        padded_mask[i, :len(m)] = m

    return padded_x, padded_y, padded_mask, embedding_processed

def get_dataloader(tokenizer, split, shuffle=True):
    dataset = HFDataset(tokenizer, split)
    pad_id = tokenizer.pad_token_id

    modified_collate_fn = lambda batch: collate_fn(batch, pad_id=pad_id)

    return DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle,
        collate_fn=modified_collate_fn, num_workers=0, pin_memory=(cfg.DEVICE != "mps"),
    )
