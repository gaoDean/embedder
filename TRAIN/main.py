import torch
from model_patcher import load_model
from dataset import get_dataloader
import config
import time

def train():
    device = config.DEVICE

    tokenizer, model = load_model()
    model.to(device)
    print(f"Model loaded")

    train_loader = get_dataloader(tokenizer, split="train")
    eval_loader = get_dataloader(tokenizer, split="test")
    print(f"Train: {len(train_loader.dataset):,}, Eval: {len(eval_loader.dataset):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    model.train()
    step = 0
    best_eval = float("inf")
    losses = []
    t0 = time.time()

    print("TRAINING")
    print("-" * 56)


def main():

#    # Test coherence
#    prompt = "Once upon a time,"
#    inputs = tokenizer(prompt, return_tensors="pt").to(device)
#
#    print(f"\nPrompt: {prompt}")
#
#    # 1. Generate without context_vector
#    with torch.no_grad():
#        out_none = model.generate(**inputs, max_new_tokens=20, do_sample=False)
#    print(f"Generated (no context): {tokenizer.decode(out_none[0], skip_special_tokens=True)}")
#
#    # 2. Generate with random context_vector
#    vec = torch.randn(1, config.CONTEXT_DIM).to(device=device, dtype=model.dtype)
#    with torch.no_grad():
#        out_vec = model.generate(**inputs, max_new_tokens=20, do_sample=False, context_vector=vec)
#    print(f"Generated (with context): {tokenizer.decode(out_vec[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    main()
