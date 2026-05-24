import torch
from model_patcher import load_model
from dataset import get_dataloader
import config as cfg
import time
import glob
import gc
import math
import resource
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

@torch.no_grad()
def evaluate(model, dataloader):
    device = cfg.DEVICE

    model.eval()
    total_loss = 0
    total_rand_e_loss = 0

    for n, (x, y, mask, e) in enumerate(dataloader):
        if n >= cfg.MAX_EVAL_TESTS:
            break

        x, y, mask, e = x.to(device), y.to(device), mask.to(device), e.to(device)

        rand_e = torch.randn(len(mask), cfg.CONTEXT_DIM).to(device=device)

        output = model(
            input_ids=x,
            labels=y,
            attention_mask=mask,
            context_vector=e
        )

        # randomise e to test if the model really does look at the vector
        rand_e_output = model(
            input_ids=x,
            labels=y,
            attention_mask=mask,
            context_vector=rand_e
        )

        loss = output.loss
        rand_e_loss = rand_e_output.loss

        total_rand_e_loss += rand_e_loss.item()
        total_loss += loss.item()

    model.train()
    return (total_loss / max(1, n)), (total_rand_e_loss / max(1, n))


def train():
    device = cfg.DEVICE


    tokenizer, model = load_model()
    model.float()

    model.to(device)
    print(f"Model loaded")

    train_loader = get_dataloader(tokenizer, split="train")
    eval_loader = get_dataloader(tokenizer, split="test", shuffle=False)
    print(f"Train: {len(train_loader.dataset):,}, Eval: {len(eval_loader.dataset):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else torch.amp.GradScaler("mps")

    model.train()
    start_step = 0
    start_epoch = 0
    best_eval = float("inf")
    losses = []
    rand_el = 0
    t0 = time.time()

    latest_cp = checkpoints.get_latest_checkpoint()

    if latest_cp:
        print(f"Resuming from full checkpoint: {latest_cp}")
        checkpoint = torch.load(latest_cp, map_location=cfg.DEVICE)
        
        # In case an old compiled checkpoint is loaded
        state_dict = checkpoint['model_state_dict']
        uncompiled_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                
        model.load_state_dict(uncompiled_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_step = checkpoint['step'] + 1
        start_epoch = checkpoint['epoch']
        best_eval = checkpoint['best_eval']
        avg = None
        elapsed = None

    if cfg.COMPILE:
        print("compiling the model...")
        # unoptimized_model = model
        model = torch.compile(model)


    if cfg.WANDB_EN:
        checkpoints.wandb_init()

    print("TRAINING")
    print(f"{'Step':>6} | {'LR':>10} | {'Train':>10} | {'Eval':>10} | {'Time':>8}")
    print("-" * 56)

    for epoch in range(start_epoch, cfg.EPOCHS):
        # ref, target, mask, embedding
        for step, (x, y, mask, e) in enumerate(train_loader, start=start_step):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            e = e.to(device)

            lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            if use_amp:
                loss = None
                with torch.amp.autocast("cuda"):
                    output = model(
                        input_ids=x,
                        labels=y,
                        attention_mask=mask,
                        context_vector=e
                    )
                    loss = output.loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(
                    input_ids=x,
                    labels=y,
                    attention_mask=mask,
                    context_vector=e
                )
                loss = output.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                optimizer.step()


            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

            if step % cfg.LOG_ITERS == 0:
                avg = sum(losses[-100:]) / len(losses[-100:])
                elapsed = time.time() - t0
                # On macOS, ru_maxrss is in bytes. Convert to MB.
                mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
                print(f"{step:6d} | {lr:10.6f} | {avg:10.4f} | {'--':>10} | {elapsed:7.1f}s | Mem: {mem_mb:.1f} MB")

            if step > 0 and step % cfg.EVAL_ITERS == 0:
                el, rand_el = evaluate(model, eval_loader)
                avg = sum(losses[-cfg.EVAL_ITERS:]) / min(len(losses), cfg.EVAL_ITERS)
                elapsed = time.time() - t0
                print(f"{step:6d} | {lr:10.6f} | {avg:10.4f} | {el:10.4f} | {elapsed:7.1f}s")

                if el < best_eval:
                    best_eval = el
                    print(f"  -> Best model (eval={el:.4f})")

            checkpoints.wandb_log(step, lr, avg, elapsed, best_eval, rand_el)

            if step > 0 and step % cfg.SAVE_ITERS == 0:
                # Save the uncompiled model state dict to avoid _orig_mod prefix
                model_sd = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
                
                checkpoint = {
                    'step': step,
                    'epoch': epoch,
                    'model_state_dict': model_sd,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_eval': best_eval
                }
                upload = step % cfg.UPLOAD_ITERS == 0
                checkpoints.save_checkpoint(checkpoint, upload=upload)


def main():
    torch.manual_seed(0)
    torch.set_float32_matmul_precision('high')

    train()

if __name__ == "__main__":
    main()
