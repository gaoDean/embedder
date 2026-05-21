import torch
from model_patcher import load_model
from dataset import get_dataloader
import config as cfg
import time
import glob
from tqdm import tqdm
import checkpoints

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < cfg.WARMUP_ITERS:
        return cfg.LEARNING_RATE * (it + 1) / (cfg.WARMUP_ITERS + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > cfg.LR_DECAY_ITERS:
        return cfg.MIN_LR
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - cfg.WARMUP_ITERS) / (cfg.LR_DECAY_ITERS - cfg.WARMUP_ITERS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return cfg.MIN_LR + coeff * (cfg.LEARNING_RATE - cfg.MIN_LR)

def train():
    device = cfg.DEVICE

    torch.manual_seed(0)

    tokenizer, model = load_model()
    model.to(device)
    print(f"Model loaded")

    train_loader = get_dataloader(tokenizer, split="train")
    eval_loader = get_dataloader(tokenizer, split="validation", shuffle=False)
    print(f"Train: {len(train_loader.dataset):,}, Eval: {len(eval_loader.dataset):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    model.train()
    start_epoch = 0
    best_eval = float("inf")
    losses = []
    t0 = time.time()

    latest_cp = checkpoints.get_latest_checkpoint()

    if latest_cp:
        print(f"Resuming from full checkpoint: {latest_cp}")
        checkpoint = torch.load(latest_cp, map_location=cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_eval = checkpoint['best_eval']

    if cfg.COMPILE:
        print("compiling the model...")
        unoptimized_model = model
        model = torch.compile(model)

    print("TRAINING")
    print("-" * 56)

    for epoch in range(start_epoch, cfg.EPOCHS):
        model.train()
        optimiser.zero_grad()

        num_steps = 0

        for step, (x, y) in enumerate(tqdm(train_loader, total=steps_per_epoch)):
            if step >= steps_per_epoch:
                break

            x, y = x.to(device), y.to(device)



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
#    vec = torch.randn(1, cfg.CONTEXT_DIM).to(device=device, dtype=model.dtype)
#    with torch.no_grad():
#        out_vec = model.generate(**inputs, max_new_tokens=20, do_sample=False, context_vector=vec)
#    print(f"Generated (with context): {tokenizer.decode(out_vec[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    main()
