import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch import autocast
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from datasets import load_dataset
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, AutoTokenizer, AutoModel
import random
import hydra
from omegaconf import DictConfig
import wandb
import tqdm
import os
import glob
import re




# https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md
class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, norm_layer=nn.BatchNorm1d):
        super().__init__()
        layers = []
        last_dim = in_dim

        # Build the hidden layers with normalization and activation
        for h_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(last_dim, h_dim))
            if norm_layer is not None:
                layers.append(norm_layer(h_dim))
            layers.append(nn.GELU()) # GELU is standard for Transformers
            last_dim = h_dim

        # Final projection layer (typically without norm/activation)
        layers.append(nn.Linear(last_dim, hidden_dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()

        self.backbone = AutoModel.from_pretrained('jinaai/jina-embeddings-v5-text-nano', trust_remote_code=True)

        self.proj = MLP(768, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

        # Cache pad_id to avoid checking on every forward pass
        self.pad_id = getattr(self.backbone.config, 'pad_token_id', None)
        if self.pad_id is None:
            self.pad_id = getattr(self.backbone.config, 'eos_token_id', 151645)

    # def forward(self, x):
    #     N, V = x.shape[:2]
    #     emb = self.backbone(x.flatten(0, 1))
    #     return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)

    def forward(self, input_ids, attention_mask=None):
        N, V = input_ids.shape[:2]

        # flatten N and V to process through roberta
        # shape becomes [N*V, Seq_Len]
        input_ids = input_ids.flatten(0, 1)

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).long()
        else:
            attention_mask = attention_mask.flatten(0, 1)

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Last-token pooling
        sequence_lengths = attention_mask.sum(dim=1) - 1
        cls_embed = outputs.last_hidden_state[torch.arange(outputs.last_hidden_state.shape[0], device=outputs.last_hidden_state.device), sequence_lengths] # shape [N*V, hidden_dim]

        z = self.proj(cls_embed) # shape [N*V, proj_dim]
        z = z.reshape(N, V, -1).transpose(0, 1) # shape [V, N, proj_dim]

        return cls_embed, z

class HFDataset(Dataset):
    def __init__(self, split="train", V=8, V_global=1, V_local=8, mask_prob=0.25):
        self.V = V
        self.V_global = V_global
        self.V_local = V_local

        self.global_len = 256
        self.local_len_max = 80 # roughly the size of a long sentence
        self.mask_prob = mask_prob

        # Initialize tokenizer once
        self.tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v5-text-nano', trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load and Pre-tokenize the dataset using Arrow/Map
        ds = load_dataset("ms_marco", "v2.1", split=split)

        def tokenize_function(examples):
            # MS Marco has lists of passages, we can just join them or pick the first for the map
            # For exact parity with your random.choice, you can flatten or just pre-tokenize the queries/passages
            # Here we tokenize the query as a fallback, and the first passage (or join them)
            texts = []
            for q, p in zip(examples["query"], examples["passages"]):
                if len(p["passage_text"]) > 0:
                    texts.append(p["passage_text"][0])
                else:
                    texts.append(q)
            return self.tokenizer(texts, add_special_tokens=True, truncation=False)

        # map caches to disk, runs heavily parallelized in C++ (Rust in fast tokenizers)
        self.ds = ds.map(
            tokenize_function,
            batched=True,
            num_proc=os.cpu_count(),
            remove_columns=ds.column_names # Drop raw text to save memory
        )
        self.ds.set_format(type="torch", columns=["input_ids"])

    def _random_crop(self, tokens, target_len):
        """Simulates an image 'crop' by taking a random contiguous span of tokens."""
        if len(tokens) <= target_len:
            return tokens

        # Start from index 0 or 1 depending on whether we want to keep a special token
        # For last-token pooling (Qwen), we shouldn't necessarily force a [CLS] token at start.
        start_idx = random.randint(0, len(tokens) - target_len)
        crop = tokens[start_idx : start_idx + target_len]

        # Add EOS token at the end if we cropped it out, since Qwen models use EOS token for last-token pooling
        if crop[-1] != self.tokenizer.eos_token_id:
            crop[-1] = self.tokenizer.eos_token_id

        return crop

    def _apply_masking(self, tokens):
        """Simulates 'color jitter/blur' by masking random tokens."""
        masked_tokens = tokens.clone()

        # Create a probability matrix for masking
        prob_matrix = torch.full(masked_tokens.shape, self.mask_prob)

        # Do not mask special tokens
        special_tokens_mask = [
            1 if t in self.tokenizer.all_special_ids else 0 for t in masked_tokens.tolist()
        ]
        prob_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        # Sample tokens to mask and replace them with a generic masking strategy
        # since this tokenizer (Qwen based) may not have a dedicated [MASK] token
        masked_indices = torch.bernoulli(prob_matrix).bool()

        # If there's no mask_token_id, we can fall back to pad_token_id or random tokens
        mask_id = self.tokenizer.mask_token_id if self.tokenizer.mask_token_id is not None else self.tokenizer.pad_token_id
        masked_tokens[masked_indices] = mask_id
        return masked_tokens

    def __getitem__(self, i):
        # Fetch pre-tokenized integers directly from zero-copy Arrow memory
        tokens = self.ds[i]["input_ids"]

        global_tokens = self._random_crop(tokens, self.global_len)

        # Pad global view to exactly 512 tokens so batches can stack
        # Qwen handles padding on the left or right, but since we rely on last-token pooling,
        # we need to be careful with attention masks.
        pad_len = self.global_len - len(global_tokens)
        if pad_len > 0:
            global_tokens = torch.cat([global_tokens, torch.full((pad_len,), self.tokenizer.pad_token_id)])

        global_views = [global_tokens for _ in range(self.V_global)]

        local_views = []
        for _ in range(self.V_local):
            # Pick a random "sentence" length (e.g., between 16 and 64 tokens)
            local_crop_len = random.randint(16, self.local_len_max)

            # Crop a small segment from the original document
            local_tokens = self._random_crop(tokens, local_crop_len)

            # Apply Masking (Corruption)
            local_tokens = self._apply_masking(local_tokens)

            # Pad local view to local_len_max (64) so local views can stack uniformly
            pad_len_local = self.local_len_max - len(local_tokens)
            if pad_len_local > 0:
                local_tokens = torch.cat([local_tokens, torch.full((pad_len_local,), self.tokenizer.pad_token_id)])

            local_views.append(local_tokens)

        return torch.stack(global_views), torch.stack(local_views)

    def __len__(self):
        return len(self.ds)

@hydra.main(version_base=None)
def main(cfg: DictConfig):
    device = ""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg, False)
    cfg.device = device

    wandb.init(project="LeJEPA embedder", config=dict(cfg))
    torch.manual_seed(0)

    train_ds = HFDataset("train", V=cfg.V)
    test_ds = HFDataset("validation", V=1)

    # Load tokenizer for logging
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v5-text-nano', trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train = DataLoader(
        train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True, num_workers=8, pin_memory=True
    )
    test = DataLoader(test_ds, batch_size=256, num_workers=4, pin_memory=True)

    net = Encoder(proj_dim=cfg.proj_dim).to(cfg.device)
    # Disable torch.compile on mps to avoid inductor error
    if hasattr(torch, "compile") and cfg.device != "mps":
        net = torch.compile(net)
    sigreg = SIGReg().to(cfg.device)
    g = { "params": net.parameters(), "lr":cfg.lr, "weight_decay": 5e-2}
    optimiser = torch.optim.AdamW([g])

    accum_steps = getattr(cfg, "accum_steps", 4)
    steps_per_epoch = getattr(cfg, "steps_per_epoch", 1000)

    # Adjust scheduler steps since we only step the optimizer every `accum_steps`
    warmup_steps = max(1, steps_per_epoch // accum_steps)
    total_steps = warmup_steps * cfg.epochs

    s1 = LinearLR(optimiser, start_factor=0.01, total_iters = warmup_steps)
    s2 = CosineAnnealingLR(optimiser, T_max=total_steps - warmup_steps, eta_min=1e-3)

    scheduler = SequentialLR(optimiser, schedulers=[s1, s2], milestones=[warmup_steps])

    # Only enable scaler if using CUDA and float16 (though you are using bfloat16,
    # we'll keep it conditionally enabled for CUDA just in case)
    scaler_enabled = (device == "cuda")
    scaler = GradScaler("cuda" if scaler_enabled else "cpu", enabled=scaler_enabled)

    import hydra.utils
    orig_cwd = hydra.utils.get_original_cwd()

    start_epoch = 0
    new_checkpoints = glob.glob(os.path.join(orig_cwd, "checkpoint_epoch_*.pt"))
    old_checkpoints = glob.glob(os.path.join(orig_cwd, "roberta_lejepa_epoch_*.pt"))

    if new_checkpoints:
        latest_cp = max(new_checkpoints, key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))
        print(f"Resuming from full checkpoint: {latest_cp}")
        checkpoint = torch.load(latest_cp, map_location=cfg.device)
        net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # Intentionally NOT loading optimizer and scheduler to start with a fresh learning rate
        # optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    elif old_checkpoints:
        latest_cp = max(old_checkpoints, key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))
        print(f"Found old format checkpoint: {latest_cp}")
        state_dict = torch.load(latest_cp, map_location=cfg.device)
        net.load_state_dict(state_dict)
        start_epoch = int(re.search(r'epoch_(\d+)', latest_cp).group(1)) + 1
        print("Note: Optimizer/Scheduler states missing in old format, restarting them.")

    for epoch in range(start_epoch, cfg.epochs):
        net.train()
        optimiser.zero_grad(set_to_none=True)

        epoch_sigreg_loss = 0.0
        num_steps = 0

        for step, (global_views, local_views) in enumerate(tqdm.tqdm(train, total=steps_per_epoch)):
            if step >= steps_per_epoch:
                break

            if step % 10 == 0:
                import html
                # Decode the first sample's first global and local view
                sample_global = global_views[0, 0].cpu().tolist()
                sample_local = local_views[0, 0].cpu().tolist()

                decoded_global = html.escape(tokenizer.decode(sample_global))
                decoded_local = html.escape(tokenizer.decode(sample_local))

                wandb.log({
                    "sample/global_view": wandb.Html(f"<pre>{decoded_global}</pre>"),
                    "sample/local_view": wandb.Html(f"<pre>{decoded_local}</pre>"),
                }, commit=False)

            if device == "cuda" or device == "mps":
                global_views = global_views.to(cfg.device, non_blocking=True).contiguous()
                local_views = local_views.to(cfg.device, non_blocking=True).contiguous()

                with autocast(device_type=device, dtype=torch.bfloat16):
                    _, proj_global = net(global_views)
                    _, proj_local = net(local_views)

                    inv_loss = (proj_global.mean(0) - proj_local).square().mean()
                    proj = torch.cat([proj_global, proj_local], dim=0)
                    sigreg_loss = sigreg(proj)

                    loss = (sigreg_loss * cfg.lam + inv_loss * (1 - cfg.lam)) / accum_steps

                scaler.scale(loss).backward()
            else:
                global_views = global_views.to(cfg.device, non_blocking=True).contiguous()
                local_views = local_views.to(cfg.device, non_blocking=True).contiguous()

                _, proj_global = net(global_views)
                _, proj_local = net(local_views)

                inv_loss = (proj_global.mean(0) - proj_local).square().mean()
                proj = torch.cat([proj_global, proj_local], dim=0)
                sigreg_loss = sigreg(proj)

                loss = (sigreg_loss * cfg.lam + inv_loss * (1 - cfg.lam)) / accum_steps
                loss.backward()

            # Update weights only after accumulating enough gradients
            if (step + 1) % accum_steps == 0 or (step + 1) == steps_per_epoch:
                if device == "cuda" or device == "mps":
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    optimiser.step()

                optimiser.zero_grad(set_to_none=True)
                scheduler.step()

            if 'sigreg_loss' in locals():
                epoch_sigreg_loss += sigreg_loss.item()
                num_steps += 1

                wandb.log(
                    {
                        "train/lejepa": loss.item(),
                        "train/sigreg": sigreg_loss.item(),
                        "train/inv": inv_loss.item(),
                        "train/lr": optimiser.param_groups[0]["lr"]
                    }
                )

        avg_sigreg = epoch_sigreg_loss / num_steps if num_steps > 0 else 0.0
        wandb.log({"train/epoch_avg_sigreg": avg_sigreg, "epoch": epoch}, commit=False)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict()
        }
        torch.save(checkpoint, os.path.join(orig_cwd, f"checkpoint_epoch_{epoch}.pt"))

    wandb.finish()

if __name__ == "__main__":
    main()
