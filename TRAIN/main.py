import torch
from model_patcher import load_model
from dataset import get_dataloader
import cfg
import time
import glob

def train():
    device = cfg.DEVICE

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

    steps_per_epoch_effective = cfg.STEPS_PER_EPOCH // cfg.GRADIENT_ACCUMULATION_STEPS
    total_steps = steps_per_epoch_effective * cfg.EPOCHS
    warmup_steps = max(1, int(steps_per_epoch_effective))

    s1 = LinearLR(optimiser, start_factor=1e-2, total_iters = warmup_steps)
    s2 = CosineAnnealingLR(optimiser, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimiser, schedulers=[s1, s2], milestones=[warmup_steps])

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    model.train()
    start_epoch = 0
    best_eval = float("inf")
    losses = []
    t0 = time.time()

    checkpoints = glob.glob(os.path.join(orig_cwd, "checkpoint_epoch_*.pt"))

    if checkpoints:
        latest_cp = max(checkpoints, key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))
        print(f"Resuming from full checkpoint: {latest_cp}")
        checkpoint = torch.load(latest_cp, map_location=cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1





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
#    vec = torch.randn(1, cfg.CONTEXT_DIM).to(device=device, dtype=model.dtype)
#    with torch.no_grad():
#        out_vec = model.generate(**inputs, max_new_tokens=20, do_sample=False, context_vector=vec)
#    print(f"Generated (with context): {tokenizer.decode(out_vec[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    main()
