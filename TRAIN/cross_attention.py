import torch
import torch.nn as nn

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
        # nn.init.normal_(self.out.weight, mean=0, std=1e-4)
        # nn.init.normal_(self.out.bias, mean=0, std=1e-4)

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
