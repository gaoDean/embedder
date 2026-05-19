import torch
from transformers import AutoTokenizer

def align_token_costs(text, jina_costs_1d, pythia_tokenizer, jina_tokenizer, device):
    """
    Maps costs from Jina tokens to Pythia tokens using character offsets.
    Optimized O(N+M) two-pointer approach, avoiding host-to-device transfers in inner loops.
    """
    # Tokenize Pythia without special tokens to match causal LM sequence
    p_enc = pythia_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    
    # Tokenize Jina WITH special tokens so j_idx aligns perfectly with the Jina model's output
    j_enc = jina_tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    
    p_offsets = p_enc['offset_mapping']
    j_offsets = j_enc['offset_mapping']
    
    # Move costs to CPU once to avoid .item() syncs in the loop. 
    # jina_costs_1d is expected to be a 1D tensor for a single sequence.
    j_costs_flat = jina_costs_1d.view(-1).tolist() 
    
    aligned_costs = []
    
    j = 0
    j_len = len(j_offsets)
    last_valid_cost = 0.0 # Fallback for unmapped tokens
    
    for p_start, p_end in p_offsets:
        if p_start == p_end: 
            aligned_costs.append(last_valid_cost)
            continue
            
        overlapping_costs = []
        
        # Advance Jina pointer to the first potentially overlapping token
        while j < j_len and j_offsets[j][1] <= p_start:
            j += 1
            
        # Temporarily search forward in Jina tokens for overlaps
        temp_j = j
        while temp_j < j_len and j_offsets[temp_j][0] < p_end:
            j_start, j_end = j_offsets[temp_j]
            # Handle standard overlapping text tokens
            if j_start != j_end and max(p_start, j_start) < min(p_end, j_end):
                if temp_j < len(j_costs_flat):
                    overlapping_costs.append(j_costs_flat[temp_j])
            # Explicitly capture special tokens (like [CLS]) if they sit exactly at the boundary
            elif j_start == j_end and (j_start == p_start or j_start == p_end):
                 if temp_j < len(j_costs_flat):
                     overlapping_costs.append(j_costs_flat[temp_j])
            temp_j += 1
        
        if overlapping_costs:
            # Pure python math, no GPU sync
            avg_cost = sum(overlapping_costs) / len(overlapping_costs)
            aligned_costs.append(avg_cost)
            last_valid_cost = avg_cost
        else:
            aligned_costs.append(last_valid_cost)
            
    return torch.tensor(aligned_costs, device=device)

import sys
print("Loading tokenizers...")
try:
    pythia_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    jina_tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
except Exception as e:
    print(f"Skipping tokenizer load: {e}")
    sys.exit(0)

text = "This is a simple test sentence to align."
j_enc = jina_tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
j_costs = torch.rand(len(j_enc['input_ids']))

print(f"Jina len: {len(j_enc['input_ids'])}")
out = align_token_costs(text, j_costs, pythia_tokenizer, jina_tokenizer, "cpu")
print(f"Pythia out len: {len(out)}")

