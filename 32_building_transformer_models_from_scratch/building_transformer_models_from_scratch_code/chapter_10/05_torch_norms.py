import torch
import torch.nn as nn

# PyTorch's LayerNorm
batch_size, seq_len, hidden_dim = 2, 5, 8
x = torch.randn(batch_size, seq_len, hidden_dim)

# LayerNorm normalizes over the last dimension
layer_norm = nn.LayerNorm(hidden_dim)
output_ln = layer_norm(x)

# RMSNorm normalizes over the last dimension
rms_norm = nn.RMSNorm(hidden_dim)
output_rms = rms_norm(x)
