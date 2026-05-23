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

@torch.no_grad()
def evaluate(model, dataloader, max_tests=50):
    device = cfg.DEVICE

    model.eval()
    total_loss = 0

    for n, (x, y, mask, e) in enumerate(dataloader):
        if n >= max_tests:
            break

        x, y, mask, e = x.to(device), y.to(device), mask.to(device), e.to(device)

        output = model(
            x,
            attention_mask=mask,
            labels=y,
            context_vector=e
        )
        loss = output.loss

        total_loss += loss.item()

    model.train()
    return total_loss / max(1, n)


def train():
    device = cfg.DEVICE

    torch.manual_seed(0)

    tokenizer, model = load_model()

    model.float()


    model.to(device)
    print(f"Model loaded")

    # train_loader = get_dataloader(tokenizer, split="train") # TODO
    # eval_loader = get_dataloader(tokenizer, split="validation", shuffle=False) # TODO
    train_loader = get_dataloader(tokenizer, split="train[:100]")
    eval_loader = get_dataloader(tokenizer, split="validation[:100]", shuffle=False)
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
    t0 = time.time()

    latest_cp = checkpoints.get_latest_checkpoint()

    if latest_cp:
        print(f"Resuming from full checkpoint: {latest_cp}")
        checkpoint = torch.load(latest_cp, map_location=cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_step = checkpoint['step'] + 1
        start_epoch = checkpoint['epoch']
        best_eval = checkpoint['best_eval']

    if cfg.COMPILE:
        print("compiling the model...")
        unoptimized_model = model
        model = torch.compile(model)

    print("TRAINING")
    print(f"{'Step':>6} | {'LR':>10} | {'Train':>10} | {'Eval':>10} | {'Time':>8}")
    print("-" * 56)

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(start_epoch, cfg.EPOCHS):
        # ref, target, mask, embedding
        for step, (x, y, mask, e) in enumerate(train_loader, start=start_step):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            e = e.to(device)


            # print(torch.mean(e))
            # print(torch.std(e))

            # vec = torch.randn(1, cfg.CONTEXT_DIM).to(device=device, dtype=model.dtype)
            # with torch.no_grad():
            #     out_vec = model.generate(x, max_new_tokens=20, do_sample=False, context_vector=vec)
            #     print(out_vec)

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
                print("LOTIGS", output)
                loss = output.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                optimizer.step()


            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

            if step % cfg.LOG_ITERS == 0:
                avg = sum(losses[-100:]) / len(losses[-100:])
                elapsed = time.time() - t0
                print(f"{step:6d} | {lr:10.6f} | {avg:10.4f} | {'--':>10} | {elapsed:7.1f}s")

            # if step > 0 and step % cfg.EVAL_ITERS == 0:
            #     el = evaluate(model, eval_loader)
            #     avg_train = sum(losses[-cfg.EVAL_ITERS:]) / min(len(losses), cfg.EVAL_ITERS)
            #     elapsed = time.time() - t0
            #     print(f"{step:6d} | {lr:10.6f} | {avg_train:10.4f} | {el:10.4f} | {elapsed:7.1f}s")
            #
            #     if el < best_eval:
            #         best_eval = el
            #         print(f"  -> Best model (eval={el:.4f})")
            #
            # if step > 0 and step % cfg.SAVE_ITERS == 0:
            #     checkpoint = {
            #         'step': step,
            #         'epoch': epoch,
            #         'model_state_dict': model_state,
            #         'optimizer_state_dict': optimizer_state,
            #         'scheduler_state_dict': scheduler_state,
            #         'scaler_state_dict': scaler_state
            #     }
            #     checkpoints.save_checkpoint(checkpoint, upload=True)

def main():
    # torch.set_default_dtype(torch.float32)
    train()

if __name__ == "__main__":
    main()

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
