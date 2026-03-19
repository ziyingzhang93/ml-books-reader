import torch
import numpy as np

def create_sinusoidal_encodings(seq_len, dim):
    N = 10000
    i = torch.arange(0, dim//2)
    div_term = torch.exp(-np.log(N) * (2*i / dim))
    position = torch.arange(seq_len).unsqueeze(1)

    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Example usage
seq_len = 512
dim = 768
positional_encodings = create_sinusoidal_encodings(seq_len, dim)
sequence = torch.randn(seq_len, dim)
sequence = sequence + positional_encodings
