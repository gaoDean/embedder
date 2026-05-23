import torch

# Model Configuration
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
CONTEXT_DIM = 1024
HIDDEN_SIZE = None  # Will be retrieved from model config if None
WANDB_EN = False

# Training Hyperparameters
WARMUP_ITERS = 2000 # from nanogpt
LR_DECAY_ITERS = 30000
LEARNING_RATE = 1e-4
MIN_LR = 6e-5
BATCH_SIZE = 20
EPOCHS = 100
GRADIENT_ACCUMULATION_STEPS = 4
GRAD_CLIP = 1.0
WEIGHT_DECAY = 5e-2 # nanogpt
DTYPE=torch.bfloat16
MAX_EVAL_TESTS=10

# Hardware & Reproducibility
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
COMPILE = False

# Logging & Checkpoints
LOG_ITERS = 20
CHECKPOINT_DIR = "checkpoints"
WANDB_PROJECT = "Inverse embedder"
DATASET_CACHE_DIR = "cache/dataset"
DATASET_ENTRY_LENGTH_LETTERS = 500 # 100 tokens * 5 letters per token
DATALOADER_BATCHSIZE = 1
EVAL_ITERS = 200
SAVE_ITERS = 10000
UPLOAD_ITERS = 100000
