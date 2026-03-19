import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # Calculate mean and variance across the last dimension(s)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm

# Example usage
batch_size, seq_len, hidden_dim = 2, 5, 128
x = torch.randn(batch_size, seq_len, hidden_dim)
layer_norm = LayerNorm()
output = layer_norm(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Output mean:\n{output.mean(axis=2)}")
print(f"Output std:\n{output.std(axis=2, correction=0)}")
