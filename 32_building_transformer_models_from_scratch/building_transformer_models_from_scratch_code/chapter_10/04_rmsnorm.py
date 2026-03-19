import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate RMS across the last dimension(s)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # Normalize
        x_norm = x * rms * self.weight
        return x_norm

# Example usage
batch_size, seq_len, hidden_dim = 2, 5, 8
x = torch.randn(batch_size, seq_len, hidden_dim)
rms_norm = RMSNorm(hidden_dim)
output = rms_norm(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Output RMS: {torch.sqrt((output**2).mean(axis=2))}")
