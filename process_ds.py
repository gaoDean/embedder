import os
import torch
import torch.nn.functional as F
import config as cfg
from datasets import load_dataset
from transformers import AutoTokenizer
import semchunk
from jina_inference import Jina

# Set num_proc back to config (default 8) to have multiple workers per GPU.
# CPU tokenization and pickling is the bottleneck, not the GPUs!
num_proc = getattr(cfg, 'DS_N_PROC', 8)

# Disable CPU parallelism in workers to prevent thread trashing across multiple processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)
num_proc_load_dataset = 8

_worker_tokenizer = None
_worker_jina_tokenizer = None
_worker_chunker = None

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
    jina_tokenizer = AutoTokenizer.from_pretrained(cfg.EMBEDDING_MODEL_NAME, trust_remote_code=True)

    if os.path.exists(cfg.DATASET_CACHE_DIR):
        print("dataset already exists")
        return None

    def process(batch, rank=0):
        """
        takes in a dataset batch

        returns {
            "input_ids": ...,
            "attention_mask": ...,
            "e_input_ids": ...,
            "e_attention_mask": ...,
        }
        """
        global _worker_tokenizer, _worker_jina_tokenizer, _worker_chunker
        if _worker_tokenizer is None:
            _worker_tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
            _worker_jina_tokenizer = AutoTokenizer.from_pretrained(cfg.EMBEDDING_MODEL_NAME, trust_remote_code=True)
            # Pass memoize=False to prevent semchunk from caching unique strings unboundedly
            _worker_chunker = semchunk.chunkerify(_worker_tokenizer, cfg.CHUNK_SIZE, memoize=False)

        tokenizer = _worker_tokenizer
        jina_tokenizer = _worker_jina_tokenizer
        chunker = _worker_chunker

        texts = batch["text"]
        chunks_nested = chunker(texts)
        chunks = [chunk for doc_chunks in chunks_nested for chunk in doc_chunks]

        import numpy as np
        
        # Sub-batch tokenization to prevent massive peak memory allocation in Rust and heap fragmentation
        sub_batch_size = 2000
        t_ids, t_masks, j_ids, j_masks = [], [], [], []
        
        for i in range(0, len(chunks), sub_batch_size):
            sub_chunks = chunks[i:i+sub_batch_size]
            
            t = tokenizer(sub_chunks, add_special_tokens=True, truncation=True, padding="max_length", max_length=cfg.CHUNK_SIZE, return_tensors="np")
            jt = jina_tokenizer(sub_chunks, add_special_tokens=True, truncation=True, padding="max_length", max_length=cfg.CHUNK_SIZE, return_tensors="np")
            
            t_ids.append(t["input_ids"])
            t_masks.append(t["attention_mask"])
            j_ids.append(jt["input_ids"])
            j_masks.append(jt["attention_mask"])
            
            del t, jt, sub_chunks
            
        out = {
            "input_ids": np.concatenate(t_ids, axis=0) if t_ids else np.array([]),
            "attention_mask": np.concatenate(t_masks, axis=0) if t_masks else np.array([]),
            "e_input_ids": np.concatenate(j_ids, axis=0) if j_ids else np.array([]),
            "e_attention_mask": np.concatenate(j_masks, axis=0) if j_masks else np.array([]),
        }
        
        import gc
        import pyarrow as pa
        import semchunk
        
        # Forcibly clear semchunk global caches in case memoize=False didn't fully work
        if hasattr(semchunk, '_memoized_token_counters'):
            semchunk._memoized_token_counters.clear()
            
        # Release PyArrow memory pool (common cause of hidden OOMs in datasets.map)
        if hasattr(pa, 'default_memory_pool'):
            pa.default_memory_pool().release_unused()
            
        del chunks, chunks_nested, texts, chunker, t_ids, t_masks, j_ids, j_masks
        gc.collect()
        
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
            process,
            batched=True,
            batch_size=cfg.DS_PROCESS_BATCH,
            writer_batch_size=cfg.DS_PROCESS_BATCH,
            remove_columns=['text'],
            desc="processing dataset",
            num_proc=num_proc,
            )

    tokenized.save_to_disk(cfg.DATASET_CACHE_DIR)

    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_folder(
        folder_path=cfg.DATASET_CACHE_DIR,
        repo_id="gaodean/openwebtext-jina",
        repo_type="dataset",
    )

    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)

if __name__ == '__main__':
    import multiprocess
    try:
        multiprocess.set_start_method('spawn')
    except RuntimeError:
        pass  # Context already set
    torch.set_float32_matmul_precision('high')
    main()
