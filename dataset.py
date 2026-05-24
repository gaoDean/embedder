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

    embedding_processed = embedding_batch[0].repeat(len(mask_batch), 1)

    for i, (x, y, m) in enumerate(zip(ref_toks_batch, target_toks_batch, mask_batch)):
        padded_x[i, :len(x)] = x
        padded_y[i, :len(y)] = y
        padded_mask[i, :len(m)] = m

    return padded_x, padded_y, padded_mask, embedding_processed

def get_dataloader(tokenizer, split, shuffle=True):
    dataset = HFDataset(split)
    pad_id = tokenizer.pad_token_id

    modified_collate_fn = lambda batch: collate_fn(batch, pad_id=pad_id)

    return DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle,
        collate_fn=modified_collate_fn, num_workers=cfg.DS_N_PROC, pin_memory=(cfg.DEVICE != "mps"),
    )
