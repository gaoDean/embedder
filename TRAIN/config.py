import torch

# Model Configuration
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
CONTEXT_DIM = 768
HIDDEN_SIZE = None  # Will be retrieved from model config if None

# Training Hyperparameters
WARMUP_ITERS = 2000 # from nanogpt
LR_DECAY_ITERS = 30000
LEARNING_RATE = 1e-4
MIN_LR = 6e-5
BATCH_SIZE = 8
EPOCHS = 100
STEPS_PER_EPOCH = 1000
GRADIENT_ACCUMULATION_STEPS = 4
MAX_LENGTH = 512
WEIGHT_DECAY = 1e-1 # nanogpt

# Hardware & Reproducibility
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42

# Logging & Checkpoints
LOG_INTERVAL = 10
SAVE_INTERVAL = 100
CHECKPOINT_DIR = "checkpoints"
WANDB_PROJECT = "pythia-hybrid-attention"
DATASET_CACHE_DIR = "dataset_cache"
DATASET_ENTRY_LENGTH_LETTERS = 500 # 100 tokens * 5 letters per token
DATALOADER_BATCHSIZE = 1
