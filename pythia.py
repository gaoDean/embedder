import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class CrossAttention(nn.Module):
    """
    A cross-attention layer that attends to an input context vector.
    Initialised to be neutral (zero output) at the start.
    """
    def __init__(self, hidden_size, context_dim=768):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(context_dim, hidden_size)
        self.value = nn.Linear(context_dim, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        
        # Zero-initialize the output projection to ensure initial neutrality.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        
    def forward(self, x, context_vector):
        if context_vector.dim() == 2:
            context_vector = context_vector.unsqueeze(1)
            
        q = self.query(x)
        k = self.key(context_vector)
        v = self.value(context_vector)
        
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        return self.out(out)

def load_model():
    model_name = "EleutherAI/pythia-70m-deduped"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    hidden_size = model.config.hidden_size
    context_dim = 768
    
    # 1. Inject Cross-Attention into each transformer block
    for layer in model.gpt_neox.layers:
        cross_attn = CrossAttention(hidden_size, context_dim)
        # Ensure new module matches model precision/device
        cross_attn.to(device=model.device, dtype=model.dtype)
        layer.add_module("cross_attention", cross_attn)
        
        def make_custom_forward(l):
            original_attention_forward = l.attention.forward
            def custom_attention_forward(hidden_states, attention_mask, layer_past=None, position_embeddings=None, **kwargs):
                # Execute original self-attention
                attn_output, layer_past = original_attention_forward(
                    hidden_states, 
                    attention_mask, 
                    layer_past=layer_past, 
                    position_embeddings=position_embeddings, 
                    **kwargs
                )
                
                # Retrieve context_vector from kwargs
                context_vector = kwargs.get("context_vector")
                if context_vector is not None:
                    cross_attn_out = l.cross_attention(attn_output, context_vector)
                    attn_output = attn_output + cross_attn_out
                
                return attn_output, layer_past
            return custom_attention_forward
        
        layer.attention.forward = make_custom_forward(layer)

    # 2. Patch top-level model forward
    original_model_forward = model.forward
    
    def model_forward(self, input_ids=None, attention_mask=None, position_ids=None, 
                      inputs_embeds=None, past_key_values=None, use_cache=None, 
                      labels=None, return_dict=None, context_vector=None, **kwargs):
        
        if input_ids is None and inputs_embeds is None and context_vector is not None:
            batch_size = context_vector.shape[0]
            device = context_vector.device
            input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            
        kwargs["context_vector"] = context_vector
        
        return original_model_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            labels=labels,
            return_dict=return_dict,
            **kwargs
        )

    model.forward = model_forward.__get__(model, type(model))
        
    return tokenizer, model

if __name__ == "__main__":
    tokenizer, model = load_model()
    print("Model modified for hybrid inference with zero-init cross-attention!")
    
    batch_size = 1
    vec = torch.randn(batch_size, 768).to(model.dtype)
    text = "The story begins with"
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, context_vector=vec)
    print(f"Verification logits shape: {outputs.logits.shape}")
