import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias

# Example usage
batch_size, seq_len, hidden_dim = 2, 5, 128
x = torch.randn(batch_size, seq_len, hidden_dim)
layer_norm = LayerNorm(hidden_dim)
output = layer_norm(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Output mean:\n{output.mean(axis=2)}")
print(f"Output std:\n{output.std(axis=2, correction=0)}")
