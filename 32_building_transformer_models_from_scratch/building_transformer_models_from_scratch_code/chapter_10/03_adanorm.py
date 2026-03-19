import torch
import torch.nn as nn

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

        # Adaptive parameters
        self.ada_weight = nn.Linear(dim, dim)
        self.ada_bias = nn.Linear(dim, dim)

    def forward(self, x):
        # Standard LayerNorm
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Adaptive scaling and shifting
        ada_w = self.ada_weight(x)
        ada_b = self.ada_bias(x)

        return x_norm * ada_w + ada_b


# Example usage
batch_size, seq_len, hidden_dim = 2, 5, 8
x = torch.randn(batch_size, seq_len, hidden_dim)

ada_ln = AdaptiveLayerNorm(hidden_dim)
output = ada_ln(x)
