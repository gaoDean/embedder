# Inverse Embedder

This project trains an inverse embedding model, using a modified Pythia-70m architecture to convert dense embedding vectors back into semantically accurate text.

## Setup
Ensure you have the dependencies installed:
```bash
pip install torch transformers datasets wandb
```

## Running the Training Script

The training script connects to Weights & Biases (W&B) for logging and checkpoint storage. Before running, ensure you are logged into W&B:
```bash
wandb login
```

To run the training script with default parameters:
```bash
python train.py
```

### Hyperparameters

You can configure the training run via command-line arguments:

- `--lr`: Learning rate (default: `1e-4`)
- `--epochs`: Number of training epochs (default: `10`)
- `--lm_weight`: Weight for the standard language modeling cross-entropy loss (default: `0.5`)
- `--semantic_weight`: Weight for the Jina embedding cost scaling (default: `1.0`)
- `--checkpoint_dir`: Directory to save local checkpoints (default: `./checkpoints`)

Example custom run:
```bash
python train.py --lr 5e-5 --epochs 20 --lm_weight 0.3 --semantic_weight 1.5 --checkpoint_dir ./my_checkpoints
```