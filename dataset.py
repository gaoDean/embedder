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
    def __init__(self, split):

        if not os.path.exists(cfg.DATASET_CACHE_DIR):
            print("no dataset cache found")
            return None

        print(f"Loading cached dataset from {cfg.DATASET_CACHE_DIR}...")
        self.ds = load_from_disk(cfg.DATASET_CACHE_DIR)[split]

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
    
    # Since process_ds.py already pads/truncates everything to exactly 15 tokens 
    # (cfg.TRUNC_LENGTH), all items in the batch are already perfectly sized to length 14.
    # We can just stack them directly!
    padded_x = torch.stack(ref_toks_batch)
    padded_y = torch.stack(target_toks_batch)
    padded_mask = torch.stack(mask_batch)
    
    # Fix the embedding broadcast bug: stack the unique embeddings for the batch
    embedding_processed = torch.stack(embedding_batch)

    # Replace the tokenizer's padding tokens in the target with -100 
    # so CrossEntropyLoss correctly ignores them during training
    padded_y[padded_y == pad_id] = -100

    return padded_x, padded_y, padded_mask, embedding_processed

def get_dataloader(tokenizer, split, shuffle=True):
    dataset = HFDataset(split)
    pad_id = tokenizer.pad_token_id

    modified_collate_fn = lambda batch: collate_fn(batch, pad_id=pad_id)

    return DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle,
        collate_fn=modified_collate_fn, num_workers=cfg.DS_N_PROC, pin_memory=(cfg.DEVICE != "mps"),
    )
