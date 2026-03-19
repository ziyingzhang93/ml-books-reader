import torch
import torch.nn as nn

class ExtrapolatingLearnedEncoding(nn.Module):
    def __init__(self, max_trained_len, d):
        super().__init__()
        self.max_trained_len = max_trained_len
        self.position_embeddings = nn.Embedding(max_trained_len, d)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len <= self.max_trained_len:
            # Normal case: use learned embeddings
            positions = torch.arange(seq_len, device=x.device)
            return x + self.position_embeddings(positions)
        else:
            # Extrapolation case: use interpolation
            positions = torch.arange(seq_len, device=x.device)
            # Interpolate between existing positions
            scale = (self.max_trained_len - 1) / (seq_len - 1)
            scaled_positions = positions * scale
            # Get floor and ceiling positions
            pos_floor = torch.floor(scaled_positions).long()
            pos_ceil = torch.ceil(scaled_positions).long()
            # Get weights for interpolation
            weights = (scaled_positions - pos_floor.float()).unsqueeze(-1)
            # Interpolate
            emb_floor = self.position_embeddings(pos_floor)
            emb_ceil = self.position_embeddings(pos_ceil)
            return x + (1 - weights) * emb_floor + weights * emb_ceil
