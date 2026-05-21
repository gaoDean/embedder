import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from cross_attention import CrossAttention
import config

def load_model(model_name=config.MODEL_NAME, context_dim=config.CONTEXT_DIM):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    hidden_size = model.config.hidden_size

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

    # outputs = model(x, labels=y)
    # logits = model.logits
    # loss = model.loss

    return tokenizer, model
