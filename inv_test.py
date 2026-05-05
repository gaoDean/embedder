import torch as t
import torch.nn as nn
import os
from transformers import AutoTokenizer, AutoModel
import numpy as np


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v5-text-nano', trust_remote_code=True)

        self.backbone = AutoModel.from_pretrained(
            'jinaai/jina-embeddings-v5-text-nano',
            trust_remote_code=True,
            # attn_implementation="flash_attention_2"
        )

        # Cache pad_id to avoid checking on every forward pass
        self.pad_id = getattr(self.backbone.config, 'pad_token_id', None)
        if self.pad_id is None:
            self.pad_id = getattr(self.backbone.config, 'eos_token_id', 151645)

    def forward(self, text):

        tokenized = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            padding=True,
            return_tensors="pt"
        ).to("mps")

        input_ids = tokenized.input_ids.unsqueeze(1)
        attention_mask = tokenized.attention_mask.unsqueeze(1)

        N, V = input_ids.shape[:2]

        # flatten N and V to process through
        # shape becomes [N*V, Seq_Len]
        input_ids = input_ids.flatten(0, 1)

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).long()
        else:
            attention_mask = attention_mask.flatten(0, 1)

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Last-token pooling
        sequence_lengths = attention_mask.sum(dim=1) - 1
        cls_embed = outputs.last_hidden_state[t.arange(outputs.last_hidden_state.shape[0], device=outputs.last_hidden_state.device), sequence_lengths] # shape [N*V, hidden_dim]

        return cls_embed



def cosine_sim(e1, e2):
    return t.dot(e1.flatten(), e2.flatten()) / (t.norm(e1) * t.norm(e2))

checkpoint = t.load(
    "jina_checkpoint.pt",
    mmap=True,
    map_location="mps",
    weights_only=True,
)

model = Encoder().to("mps")

state_dict = checkpoint.get('model_state_dict', checkpoint)

# Capture the output to monitor strict=False behavior
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

if missing_keys or unexpected_keys:
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

model.eval()

text = ["a person with a crown"]

captured_gradients = []
def capture_grad_hook(module, grad_input, grad_output):
    captured_gradients.append(grad_output[0].clone().detach())

captured_forward = []
def capture_forward_hook(module, forward_input, forward_output):
    captured_forward.append(forward_input[0].clone().detach())

embedding_layer = model.backbone.get_input_embeddings()
embedding_layer.weight.requires_grad_(True)
grad_hook_handle = embedding_layer.register_full_backward_hook(capture_grad_hook)
forward_hook_handle = embedding_layer.register_forward_hook(capture_forward_hook)



print(model)

model.zero_grad()
res1 = model(["queen"])
model.zero_grad()
res2 = model(["queen without a crown"])

print(res2)

# print(cosine_sim(res[0], res[1]))
# print(cosine_sim(res[1], res[2]))
# print(cosine_sim(res[0], res[3]))

# loss = ((res1[0] - res2[0]) ** 2).sum()

loss = cosine_sim(res1[0], res2[0]) ** 2

print(loss)

loss.backward()


print(captured_gradients[0].shape)
print(captured_forward[0].shape)

print(captured_gradients[0])
print(captured_gradients[0][0])

token_contributions = t.norm(captured_gradients[0][0], dim=1)

print(token_contributions)
