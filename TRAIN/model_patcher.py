import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from cross_attention import CrossAttention
import config

def load_model(model_name=config.MODEL_NAME, context_dim=config.CONTEXT_DIM):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    hidden_size = model.config.hidden_size

    for layer in model.gpt_neox.layers:
        cross_attn = CrossAttention(hidden_size, context_dim)
        cross_attn.to(device=model.device)

        layer.attention.add_module("cross_attention", cross_attn)
        layer.attention.context_vector = None

        def attention_forward_hook(module, args, output):
            attn_output = output[0]

            if module.context_vector is not None:
                cross_attn_out = module.cross_attention(attn_output, module.context_vector)
                attn_output = attn_output + cross_attn_out

            return (attn_output,) + output[1:]

        layer.attention.register_forward_hook(attention_forward_hook)

    def model_pre_hook(module, args, kwargs):
        context_vector = kwargs.pop("context_vector", None)

        if kwargs.get("input_ids") is None and kwargs.get("inputs_embeds") is None and context_vector is not None:
            batch_size = context_vector.shape[0]
            device = context_vector.device
            kwargs["input_ids"] = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

        if context_vector is not None:
            for layer in module.gpt_neox.layers:
                layer.attention.context_vector = context_vector

        return args, kwargs

    def model_post_hook(module, args, output):
        for layer in module.gpt_neox.layers:
            layer.attention.context_vector = None
        return output

    model.register_forward_pre_hook(model_pre_hook, with_kwargs=True)
    model.register_forward_hook(model_post_hook)

    return tokenizer, model
