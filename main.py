import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch import autocast
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from datasets import load_dataset
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, AutoTokenizer
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

        self.backbone = RobertaModel.from_pretrained('distilroberta-base')
        
        # DistilRoBERTa has 6 layers. We reinitialize the top 3 layers (50%) 
        # so it keeps basic language understanding but learns higher-level features from scratch.
        for layer in self.backbone.encoder.layer[3:]:
            layer.apply(self.backbone._init_weights)

        self.proj = MLP(768, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

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
            pad_id = self.backbone.config.pad_token_id
            attention_mask = (input_ids != pad_id).long()
        else:
            attention_mask = attention_mask.flatten(0, 1)

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = outputs.pooler_output # shape [N*V, 768]

        z = self.proj(cls_embed) # shape [N*V, proj_dim]
        z = z.reshape(N, V, -1).transpose(0, 1) # shape [V, N, proj_dim]

        return cls_embed, z

class HFDataset(Dataset):
    def __init__(self, split="train", V=8, V_global=1, V_local=8, mask_prob=0.15):
        self.V = V
        self.V_global = V_global
        self.V_local = V_local

        self.global_len = 256
        self.local_len_max = 48 # roughly the size of a long sentence
        self.mask_prob = mask_prob

        self.ds = load_dataset("ms_marco", "v2.1", split=split)
        # self.ds = load_dataset("JeanKaddour/minipile", split=split)

        self.tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    def _random_crop(self, tokens, target_len):
        """Simulates an image 'crop' by taking a random contiguous span of tokens."""
        if len(tokens) <= target_len:
            return tokens

        # Start from index 1 to avoid duplicating an existing [CLS] token,
        # and leave room to manually add one back
        start_idx = random.randint(1, len(tokens) - target_len)
        crop = tokens[start_idx : start_idx + target_len - 1]

        # Prepend [CLS] (token ID 0 for RoBERTa)
        cls_token = torch.tensor([self.tokenizer.cls_token_id], dtype=tokens.dtype)
        return torch.cat([cls_token, crop])

    def _apply_masking(self, tokens):
        """Simulates 'color jitter/blur' by masking random tokens."""
        masked_tokens = tokens.clone()

        # Create a probability matrix for masking
        prob_matrix = torch.full(masked_tokens.shape, self.mask_prob)

        # Do not mask special tokens (like [CLS], [SEP], [PAD])
        special_tokens_mask = [
            1 if t in self.tokenizer.all_special_ids else 0 for t in masked_tokens.tolist()
        ]
        prob_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        # Sample tokens to mask and replace them with [MASK]
        masked_indices = torch.bernoulli(prob_matrix).bool()
        masked_tokens[masked_indices] = self.tokenizer.mask_token_id
        return masked_tokens

    def __getitem__(self, i):
        row = self.ds[i]

        passages_list = row["passages"]["passage_text"]
        if len(passages_list) == 0:
            # Fallback in the extremely rare case of an empty passage list
            text = row["query"]
        else:
            text = random.choice(passages_list)

        # Tokenize the full document first (no padding/truncation yet)
        tokens = self.tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0]

        global_tokens = self._random_crop(tokens, self.global_len)

        # Pad global view to exactly 512 tokens so batches can stack
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
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    
    train = DataLoader(
        train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True, num_workers=0
    )
    test = DataLoader(test_ds, batch_size=256, num_workers=4)

    net = Encoder(proj_dim=cfg.proj_dim).to(cfg.device)
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

    scaler = GradScaler("cuda", enabled=True)

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
        optimiser.zero_grad()

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
                global_views = global_views.to(cfg.device, non_blocking=True).long()
                local_views = local_views.to(cfg.device, non_blocking=True).long()

                with autocast(device_type=device, dtype=torch.bfloat16):
                    _, proj_global = net(global_views)
                    _, proj_local = net(local_views)

                    inv_loss = (proj_global.mean(0) - proj_local).square().mean()
                    
                    proj = torch.cat([proj_global, proj_local], dim=0)
                    sigreg_loss = sigreg(proj)

                    loss = sigreg_loss * cfg.lam + inv_loss * (1 - cfg.lam)

                # Normalize the loss based on accumulation steps
                loss = loss / accum_steps
                scaler.scale(loss).backward()
            else:
                global_views = global_views.to(cfg.device, non_blocking=True).long()
                local_views = local_views.to(cfg.device, non_blocking=True).long()

                _, proj_global = net(global_views)
                _, proj_local = net(local_views)

                inv_loss = (proj_global.mean(0) - proj_local).square().mean()

                proj = torch.cat([proj_global, proj_local], dim=0)
                sigreg_loss = sigreg(proj)

                loss = sigreg_loss * cfg.lam + inv_loss * (1 - cfg.lam)

                # Normalize the loss based on accumulation steps
                loss = loss / accum_steps
                loss.backward()

            # Update weights only after accumulating enough gradients
            if (step + 1) % accum_steps == 0 or (step + 1) == steps_per_epoch:
                if device == "cuda" or device == "mps":
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    optimiser.step()
                optimiser.zero_grad()
                scheduler.step()

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
