import os
import torch
import torch.nn.functional as F
import config as cfg
from datasets import load_dataset
from transformers import AutoTokenizer
from jina_inference import Jina

# Set num_proc back to config (default 8) to have multiple workers per GPU.
# CPU tokenization and pickling is the bottleneck, not the GPUs!
num_proc = getattr(cfg, 'DS_N_PROC', 8)

# Disable CPU parallelism in workers to prevent thread trashing across multiple processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)
num_proc_load_dataset = 8

portions_buffer = []
jina = None

def get_portions(paragraph, portion_length):

    portions = []

    for i in range(1, (len(paragraph) // portion_length) + 1, 1):
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

    def process(batch, rank=0):
        """
        takes in a dataset batch

        returns {
            "input_ids": ...,
            "attention_mask": ...,
            "embeddings": ...,
        }
        """
        global portions_buffer
        global jina

        if jina is None:
            if torch.cuda.is_available():
                device = f"cuda:{rank % torch.cuda.device_count()}"
            else:
                device = cfg.DEVICE
            jina = Jina(device=device)

        unprocessed_texts = batch["text"]

        # the idea is that we split the paragraphs into "portions"
        # which are each one to two sentences long
        # so we can make better use of our dataset
        for text in unprocessed_texts:
            portions = get_portions(text, cfg.MAX_CHARS_TRUNC)
            for portion in portions:
                portions_buffer.append(portion)

        # number of portions available to process
        # lets say buffer reaches a length of 40, if DS_PROCESS_BATCH is 30, then we can process 30 portions
        # if buffer reaches length of 80, we can process 30 + 30 portions sequentially
        # this is to keep batch size constant to take advantage of compilation
        num_portions_batches = len(portions_buffer) // cfg.DS_PROCESS_BATCH
        output = None
        if num_portions_batches >= 1:
            for i in range(num_portions_batches):

                # pop the processable portions e.g. pop the first 30 entires
                to_process = portions_buffer[:cfg.DS_PROCESS_BATCH]
                portions_buffer = portions_buffer[cfg.DS_PROCESS_BATCH:]

                tokenized = tokenizer(to_process, add_special_tokens=True, truncation=True, padding="max_length", max_length=cfg.TRUNC_LENGTH)

                embeddings = None
                with torch.no_grad():
                    embeddings = F.layer_norm(jina.model(to_process, cfg.TRUNC_LENGTH), (cfg.CONTEXT_DIM,))
                
                # Cast to float16 and convert to a list of numpy arrays.
                # This prevents python's .tolist() from upcasting everything to 64-bit float objects,
                # which would cause the Hugging Face dataset to consume 4x more disk space.
                tokenized["embeddings"] = list(embeddings.detach().cpu().to(torch.float16).numpy())

                # merge output with the tokenized dict
                if output is None:
                    output = tokenized
                else:
                    output = {key: output[key] + tokenized[key] for key in tokenized}

        if output is None:
            # Hugging Face map requires a dictionary with correct keys and empty lists
            # to avoid silently reverting/ignoring the whole mapping process
            output = {key: [] for key in tokenizer([""], max_length=cfg.TRUNC_LENGTH).keys()}
            output["embeddings"] = []

        return output

    # tokenize the dataset
    tokenized = split_dataset.map(
            process,
            batched=True,
            batch_size=cfg.DS_PROCESS_BATCH,
            remove_columns=['text'],
            desc="processing dataset",
            num_proc=num_proc,
            with_rank=True,
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
