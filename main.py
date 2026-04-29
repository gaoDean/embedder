import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch import autocast
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, AutoTokenizer
import random

config = RobertaConfig.from_pretrained('roberta-base')
model_scratch = RobertaModel(config)


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

        config = RobertaConfig.from_pretrained('roberta-base')
        self.backbone = RobertaModel(config)

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
    def __init__(self, split="train", V_global=1, V_local=8, mask_prob=0.15):
        self.V_global = V_global
        self.V_local = V_local

        self.global_len = 256
        self.local_len_max = 48 # roughly the size of a long sentence
        self.mask_prob = mask_prob

        self.ds = load_dataset("ms_marco", "v2.1", split=split)
        # self.ds = load_dataset("JeanKaddour/minipile", split=split)

        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

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
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    cfg.device = device

    wandb.init(project="LeJEPA embedder", config=dict(cfg))
    torch.manual_seed(0)

    train_ds = HFDataset("train", V=cfg.V)
    test_ds = HFDataset("validation", V=1)
    train = DataLoader(
        train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True, num_workers=8
    )
    test = DataLoader(test_ds, batch_size=256, num_workers=8)

    net = Encoder(proj_dim=cfg.proj_dim).to(cfg.device)
    sigreg = SIGREG().to(cfg.device)
    g = { "params": net.parameters(), "lr":cfg.lr, "weight_decay": 5e-2}
    optimiser = torch.optim.AdamW(g)

    warmup_steps = len(train)
    total_steps =len(train) * cfg.epochs

    s1 = LinearLR(optimiser, start_factor=0.01, total_iters = warmup_steps)
    s2 = CosineAnnealingLR(optimiser, T_max=total_steps - warmup_steps, eta_min=1e-3)

    scheduler = SequentialLR(optimiser, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScalar("cuda", enabled=False)

    for epoch in range(cfg.epochs):
        net.train()

        for global_views, local_views in tqdm.tqdm(train, total=len(train)):
            with autocast(dtype=torch.bfloat16):
                global_views = global_views.to(cfg.device, non_blocking=True)
                local_views = local_views.to(cfg.device, non_blocking=True)

                views = torch.cat([global_views, local_views])

                emb, proj = net(views)

                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)

                loss = sigreg_loss * cfg.lam + inv_loss * (1 - cfg.lam)

            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optmiser)
            scaler.update()
            scheduler.step()

            wandb.log(
                {
                    "train/lejepa": loss.item(),
                    "train/sigreg": sigreg_loss.item(),
                    "train/inv": inv_loss.item(),
                    "train/lr": opt.param_groups[0]["lr"]
                }
            )


        torch.save(net.state_dict(), f"roberta_lejepa_epoch_{epoch}.pt")

    wandb.finish()

if __name__ == "__main__":
    main()
