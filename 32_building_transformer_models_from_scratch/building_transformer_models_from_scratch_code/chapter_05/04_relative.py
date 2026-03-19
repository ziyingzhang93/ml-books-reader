import torch
import torch.nn as nn

class RelativePositionalEncoding(nn.Module):
    def __init__(self, max_relative_position, d_model):
        super().__init__()
        self.max_relative_position = max_relative_position
        self.relative_attention_bias = nn.Parameter(
            torch.randn(2 * max_relative_position + 1, d_model)
        )

    def forward(self, length):
        context_position = torch.arange(length, dtype=torch.long)[:, None]
        memory_position = torch.arange(length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = relative_position + self.max_relative_position
        return self.relative_attention_bias[relative_position_bucket]
