import torch
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
        )

        # Cache pad_id to avoid checking on every forward pass
        self.pad_id = getattr(self.backbone.config, 'pad_token_id', None)
        if self.pad_id is None:
            self.pad_id = getattr(self.backbone.config, 'eos_token_id', 151645)

    def forward(self, text, return_tokens=False):
        '''
        return_tokens determines whether a second return value is outputted
        an array of tokenized strings.
        '''

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
        cls_embed = outputs.last_hidden_state[torch.arange(outputs.last_hidden_state.shape[0], device=outputs.last_hidden_state.device), sequence_lengths] # shape [N*V, hidden_dim]

        if return_tokens:
            decoded = [self.tokenizer.decode([token_id]) for token_id in input_ids[0]]

            return cls_embed, decoded

        return cls_embed

class Jina():
    def __init__(self):
        self.model = Encoder().to("mps")
        self.model.eval()

    def embed(self, text_arr, batch_size=32):
        '''
        text_arr is an array of text to encode
        '''

        all_embeddings = []
        for i in range(0, len(text_arr), batch_size):
            batch = text_arr[i : i + batch_size]
            with torch.no_grad():
                embeddings = self.model(batch)
                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)
