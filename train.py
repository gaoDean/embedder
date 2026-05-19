import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from torch.utils.data import DataLoader
from datasets import load_dataset
import tqdm

import argparse
import os
import glob
import re
import wandb
from torch.amp import autocast, GradScaler

# --- 1. Model Modification (Cross-Attention) ---
class NeutralCrossAttention(nn.Module):
    def __init__(self, hidden_size, embed_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(embed_dim, hidden_size)
        self.v_proj = nn.Linear(embed_dim, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Initialize output projection to zero for neutral start
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, hidden_states, target_embedding):
        # hidden_states: [B, SeqLen, hidden_size]
        # target_embedding: [B, 1, embed_dim]
        q = self.q_proj(hidden_states)
        k = self.k_proj(target_embedding)
        v = self.v_proj(target_embedding)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-1, -2)) / (q.size(-1) ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        # Residual connection
        return hidden_states + self.out_proj(out)

class AttentionWithCrossAttn(nn.Module):
    def __init__(self, original_attn, hidden_size, embed_dim):
        super().__init__()
        self.original_attn = original_attn
        self.cross_attn = NeutralCrossAttention(hidden_size, embed_dim)
        self.target_embedding = None # Stored temporarily during forward pass

    def forward(self, *args, **kwargs):
        # Original self-attention forward
        outputs = self.original_attn(*args, **kwargs)
        hidden_states = outputs[0]
        
        # Apply cross-attention sequentially after self-attention
        if self.target_embedding is not None:
            hidden_states = self.cross_attn(hidden_states, self.target_embedding)
            
        return (hidden_states,) + outputs[1:]

def modify_pythia(model, embed_dim=768):
    hidden_size = model.config.hidden_size
    for i, layer in enumerate(model.gpt_neox.layers):
        # Wrap only the self-attention mechanism, leaving MLP and residual logic intact
        layer.attention = AttentionWithCrossAttn(layer.attention, hidden_size, embed_dim)
    return model

def set_target_embedding(model, target_embedding):
    """Sets the target embedding for all cross-attention layers."""
    for layer in model.gpt_neox.layers:
        if isinstance(layer.attention, AttentionWithCrossAttn):
            layer.attention.target_embedding = target_embedding.unsqueeze(1) # [B, 1, embed_dim]

# --- 2. Jina Evaluator & Hooks ---
class JinaEvaluator:
    def __init__(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v5-text-nano', trust_remote_code=True)
        self.model = AutoModel.from_pretrained('jinaai/jina-embeddings-v5-text-nano', trust_remote_code=True).to(device)
        self.model.eval()
        self.device = device
        
        # Freeze Jina model to save memory
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Setup hooks to capture gradients on inputs
        self.captured_gradients = None
        embedding_layer = self.model.get_input_embeddings()
        
        # Explicitly unfreeze the embedding layer so it can receive gradients
        embedding_layer.weight.requires_grad_(True)
        
        def capture_grad_hook(module, grad_input, grad_output):
            self.captured_gradients = grad_output[0].clone().detach()
            
        self.hook = embedding_layer.register_full_backward_hook(capture_grad_hook)

    def evaluate_and_get_costs(self, text_list, target_embeddings):
        tokens = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        self.model.zero_grad()
        
        with autocast(device_type=self.device, enabled=(self.device in ["cuda", "mps"])):
            outputs = self.model(**tokens)
            
            # Last-token pooling (as in inv_test.py)
            mask = tokens.attention_mask
            seq_lengths = mask.sum(dim=1) - 1
            generated_embeddings = outputs.last_hidden_state[torch.arange(outputs.last_hidden_state.shape[0]), seq_lengths]
            
            # Calculate loss (e.g., L2 norm difference)
            loss = torch.norm(generated_embeddings - target_embeddings, dim=-1).mean()
            
        loss.backward()
        
        # Calculate token costs (gradient magnitude)
        grad_magnitudes = torch.norm(self.captured_gradients, dim=-1) # [B, SeqLen]
        
        # Normalize to create a cost (low magnitude = irrelevant = high cost)
        # We invert the magnitude: max magnitude becomes 0 cost, 0 magnitude becomes max cost
        max_grad = grad_magnitudes.max(dim=1, keepdim=True)[0] + 1e-8
        costs = 1.0 - (grad_magnitudes / max_grad) 
        
        return costs, tokens.input_ids

# --- 3. Token Alignment & Thesaurus Smoothing ---
def align_token_costs(text, jina_costs, pythia_tokenizer, jina_tokenizer, device):
    """
    Maps costs from Jina tokens to Pythia tokens using character offsets.
    """
    # Tokenize with offsets
    p_enc = pythia_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    j_enc = jina_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    
    p_offsets = p_enc['offset_mapping']
    j_offsets = j_enc['offset_mapping']
    
    # We strip the batch dimension for jina_costs in this loop, assuming batch_size=1 for now
    j_costs_flat = jina_costs[0] 
    
    aligned_costs = []
    
    for p_start, p_end in p_offsets:
        if p_start == p_end: # Special tokens might have (0,0)
            aligned_costs.append(0.0)
            continue
            
        overlapping_costs = []
        for j_idx, (j_start, j_end) in enumerate(j_offsets):
            if j_start == j_end:
                continue
            # Check for overlap between (p_start, p_end) and (j_start, j_end)
            if max(p_start, j_start) < min(p_end, j_end):
                # Ensure we don't go out of bounds (Jina might have added special tokens)
                if j_idx < len(j_costs_flat):
                    overlapping_costs.append(j_costs_flat[j_idx].item())
        
        # Average the costs of overlapping Jina tokens
        if overlapping_costs:
            aligned_costs.append(sum(overlapping_costs) / len(overlapping_costs))
        else:
            aligned_costs.append(0.0) # Fallback
            
    # Convert to tensor, add batch dimension back
    # Note: p_enc['input_ids'] length might differ slightly from the original generated sequence 
    # due to special tokens, so we handle that in the main loop.
    return torch.tensor(aligned_costs, device=device).unsqueeze(0), p_enc['input_ids']

def build_thesaurus(tokenizer, model, device):
    """Pre-computes similarity between vocabulary tokens using model's input embeddings."""
    embeddings = model.get_input_embeddings().weight.detach() # [VocabSize, Hidden]
    # Normalize for cosine similarity
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def smooth_costs(costs, input_ids, thesaurus_embeddings, pythia_vocab_size):
    """Distributes costs to similar words in the vocabulary (Vectorized)."""
    # input_ids: [B, SeqLen]
    # costs: [B, SeqLen]
    # thesaurus_embeddings: [VocabSize, Hidden]
    
    # 1. Fetch embeddings for the generated tokens: [B, SeqLen, Hidden]
    token_embs = thesaurus_embeddings[input_ids]
    
    # 2. Compute similarity against the entire vocabulary
    # [B, SeqLen, Hidden] @ [Hidden, VocabSize] -> [B, SeqLen, VocabSize]
    sims = torch.matmul(token_embs, thesaurus_embeddings.T)
    
    # 3. Apply ReLU and square to sharpen similarities
    sims = F.relu(sims) ** 2
    
    # 4. Multiply by base costs
    # costs.unsqueeze(-1) expands to [B, SeqLen, 1] for broadcasting
    smoothed_costs = costs.unsqueeze(-1) * sims
    
    return smoothed_costs

# --- 4. Main Training Loop ---
def main(args):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    wandb.init(project="Inverse-Embedder", config=vars(args))
    
    # Load Models
    print("Loading Pythia...")
    pythia_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    pythia_tokenizer.pad_token = pythia_tokenizer.eos_token
    pythia_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped").to(device)
    pythia_model = modify_pythia(pythia_model, embed_dim=768)
    
    print("Loading Jina...")
    jina_eval = JinaEvaluator(device)
    
    # Thesaurus for semantic smoothing
    thesaurus = build_thesaurus(pythia_tokenizer, pythia_model, device)
    
    optimizer = torch.optim.AdamW(pythia_model.parameters(), lr=args.lr)
    
    scaler_enabled = (device == "cuda")
    scaler = GradScaler("cuda" if scaler_enabled else "cpu", enabled=scaler_enabled)
    
    # --- Checkpointing Setup ---
    start_epoch = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoints = glob.glob(os.path.join(args.checkpoint_dir, "checkpoint_epoch_*.pt"))
    if checkpoints:
        latest_cp = max(checkpoints, key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))
        print(f"Resuming from checkpoint: {latest_cp}")
        checkpoint = torch.load(latest_cp, map_location=device)
        pythia_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    pythia_model.train()
    
    # Dummy target embeddings for illustration (in practice, load from pre-cached dataset/dataloader)
    # We simulate a dataset yielding batches of embeddings.
    
    for epoch in range(start_epoch, args.epochs):
        print(f"Starting Epoch {epoch}...")
        
        # Simulated Dataloader Loop
        for step in range(args.steps_per_epoch):
            # Simulated batch of embeddings [BatchSize, EmbedDim]
            target_embeddings = torch.randn(args.batch_size, 768).to(device) 
            
            # 1. Set target embedding for the batch
            set_target_embedding(pythia_model, target_embeddings)
            
            # 2. Generate 512 tokens (without gradients)
            with torch.no_grad(), autocast(device_type=device, enabled=(device in ["cuda", "mps"])):
                # We need a prompt for each item in the batch
                prompt_texts = ["<|endoftext|>"] * args.batch_size
                prompt = pythia_tokenizer(prompt_texts, return_tensors="pt").to(device)
                
                generated_ids = pythia_model.generate(
                    **prompt, 
                    max_new_tokens=512, 
                    pad_token_id=pythia_tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.9
                )
                generated_texts = pythia_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
            if step % 10 == 0:
                print(f"Epoch {epoch} | Step {step} | Sample Text: {generated_texts[0][:100]}...")
            
            # 3. Evaluate with Jina and get token-level costs
            costs, jina_input_ids = jina_eval.evaluate_and_get_costs(generated_texts, target_embeddings)
            
            # 4. Align Jina costs back to Pythia tokens (batch processing)
            aligned_costs_list = []
            clean_pythia_ids_list = []
            
            for b in range(args.batch_size):
                a_cost, c_ids = align_token_costs(
                    generated_texts[b], costs[b:b+1], pythia_tokenizer, jina_eval.tokenizer, device
                )
                aligned_costs_list.append(a_cost.squeeze(0)) # [SeqLen]
                clean_pythia_ids_list.append(c_ids) # List of token lists
                
            # Pad aligned sequences to match max length in batch
            max_len = max(len(ids) for ids in clean_pythia_ids_list)
            padded_aligned_costs = torch.zeros(args.batch_size, max_len, device=device)
            padded_clean_ids = torch.full((args.batch_size, max_len), pythia_tokenizer.pad_token_id, device=device)
            
            for b in range(args.batch_size):
                seq_l = len(clean_pythia_ids_list[b])
                padded_aligned_costs[b, :seq_l] = aligned_costs_list[b]
                padded_clean_ids[b, :seq_l] = torch.tensor(clean_pythia_ids_list[b], device=device)
            
            # 5. Thesaurus Smoothing
            with autocast(device_type=device, enabled=(device in ["cuda", "mps"])):
                smoothed_costs = smooth_costs(padded_aligned_costs, padded_clean_ids, thesaurus, pythia_model.config.vocab_size)
            
            # 6. Pythia Forward & Vectorized Backpropagation
            optimizer.zero_grad()
            
            with autocast(device_type=device, enabled=(device in ["cuda", "mps"])):
                outputs = pythia_model(padded_clean_ids)
                logits = outputs.logits[:, :-1, :] # Shift for next token prediction
                labels = padded_clean_ids[:, 1:]
                
                # Apply standard cross entropy
                ce_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), 
                    labels.reshape(-1), 
                    reduction='none',
                    ignore_index=pythia_tokenizer.pad_token_id
                )
                ce_loss = ce_loss.view(args.batch_size, -1) # [B, SeqLen]
                
                # Align smoothed_costs (shift by 1 to match next-token prediction)
                target_costs = smoothed_costs[:, 1:, :] # [B, SeqLen, VocabSize]
                
                # To apply the thesaurus costs, we compute a custom cross-entropy-like loss
                # where the logits are evaluated against the smoothed cost distribution.
                # Since we want to reward/penalize based on the smoothed costs, we use KL Divergence.
                # We normalize target_costs to be a valid probability distribution over the vocab.
                
                # Convert costs into a target probability distribution (lower cost = higher probability target)
                # First, ensure costs are positive
                costs_shifted = target_costs - target_costs.min(dim=-1, keepdim=True)[0]
                # Invert: max cost becomes 0, 0 cost becomes max
                inverted_costs = costs_shifted.max(dim=-1, keepdim=True)[0] - costs_shifted
                # Normalize to sum to 1
                target_probs = inverted_costs / (inverted_costs.sum(dim=-1, keepdim=True) + 1e-8)
                
                # Compute KL divergence between Pythia logits and the smoothed target distribution
                log_probs = F.log_softmax(logits, dim=-1)
                semantic_loss = F.kl_div(log_probs, target_probs, reduction='none').sum(dim=-1) # [B, SeqLen]
                
                final_loss = (args.lm_weight * ce_loss) + (args.semantic_weight * semantic_loss)
                
                # Mask out padding from final loss
                mask = (labels != pythia_tokenizer.pad_token_id).float()
                final_loss = (final_loss * mask).sum() / mask.sum() # Mean over non-padding tokens
            
            if scaler_enabled:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                final_loss.backward()
                optimizer.step()
            
            wandb.log({
                "train/loss": final_loss.item(),
                "train/ce_loss_mean": ce_loss.mean().item(),
                "train/semantic_loss_mean": semantic_loss.mean().item(),
                "epoch": epoch,
                "step": step + (epoch * args.steps_per_epoch)
            })
        
        # --- Checkpoint Saving ---
        checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': pythia_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        
        # Upload artifact to W&B
        artifact = wandb.Artifact(f"model-checkpoint-epoch-{epoch}", type="model")
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Inverse Embedder")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per step")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="Steps per epoch (simulating a dataloader length)")
    parser.add_argument("--lm_weight", type=float, default=0.5, help="Weight for language modeling loss")
    parser.add_argument("--semantic_weight", type=float, default=1.0, help="Weight for semantic/Jina loss")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    
    args = parser.parse_args()
    main(args)
