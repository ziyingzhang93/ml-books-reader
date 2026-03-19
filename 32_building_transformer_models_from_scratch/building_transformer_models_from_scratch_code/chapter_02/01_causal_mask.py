import torch

seq_len = 4
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
print(causal_mask)
