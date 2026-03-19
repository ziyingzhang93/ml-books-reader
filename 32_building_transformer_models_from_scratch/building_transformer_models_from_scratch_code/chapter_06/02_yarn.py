import torch
import torch.nn as nn
import numpy as np

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

class YaRN(nn.Module):
    def __init__(self, dim, orig_seq_len=512, scale=4, alpha=1, beta=32):
        super().__init__()
        N = 10000
        pos_freq = N ** (torch.arange(0, dim, 2).float() / dim)
        inv_freq_extrapolation = 1. / pos_freq
        inv_freq_interpolation = 1. / (scale * pos_freq)

        low = dim * np.log(orig_seq_len / (2*np.pi*beta)) / (2*np.log(N))
        high = dim * np.log(orig_seq_len / (2*np.pi*alpha)) / (2*np.log(N))
        low = max(np.floor(low), 0)
        high = min(np.ceil(high), dim-1)

        linear_func = (torch.arange(dim // 2).float() - low) / (high - low)
        ramp_func = torch.clamp(linear_func, 0, 1)
        inv_freq_factor = 1 - ramp_func
        inv_freq = inv_freq_interpolation * (1-inv_freq_factor) + \
                   inv_freq_extrapolation * inv_freq_factor

        # Original RoPE multiplied with a scaling factor
        scaling_factor = 0.1 * np.log(scale) + 1.0
        position = torch.arange(orig_seq_len * scale).float()
        sinusoid_inp = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos() * scaling_factor)
        self.register_buffer("sin", sinusoid_inp.sin() * scaling_factor)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)
        cos = self.cos[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin[:seq_len].view(1, seq_len, 1, -1)
        return apply_rotary_pos_emb(x, cos, sin)
