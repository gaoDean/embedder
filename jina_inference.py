import torch
import torch.nn as nn
import os
from transformers import AutoTokenizer, AutoModel
import numpy as np
import config as cfg


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.EMBEDDING_MODEL_NAME, trust_remote_code=True)

        self.backbone = AutoModel.from_pretrained(
            cfg.EMBEDDING_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=cfg.DTYPE_HALF,
        )

        # Cache pad_id to avoid checking on every forward pass
        self.pad_id = getattr(self.backbone.config, 'pad_token_id', None)
        if self.pad_id is None:
            self.pad_id = getattr(self.backbone.config, 'eos_token_id', 151645)

    def forward(self, text, trunc_length=None):
        '''
        return_tokens determines whether a second return value is outputted
        an array of tokenized strings.
        '''


        tokenized = None
        if trunc_length:
            tokenized = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                max_length=trunc_length,
                return_tensors="pt"
            ).to(cfg.DEVICE)
        else:
            tokenized = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                padding=True,
                return_tensors="pt"
            ).to(cfg.DEVICE)

        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).long()

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Last-token pooling
        sequence_lengths = attention_mask.sum(dim=1) - 1
        cls_embed = outputs.last_hidden_state[torch.arange(outputs.last_hidden_state.shape[0], device=outputs.last_hidden_state.device), sequence_lengths] # shape [N, hidden_dim]

        if return_tokens:
            decoded = [self.tokenizer.decode([token_id]) for token_id in input_ids[0]]

            return cls_embed, decoded

        return cls_embed

class Jina():
    def __init__(self):
        self.model = Encoder().to(cfg.DEVICE)
        self.model.eval()
