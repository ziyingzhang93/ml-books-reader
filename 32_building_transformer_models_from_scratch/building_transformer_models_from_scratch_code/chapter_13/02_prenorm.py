import torch.nn as nn

class PreNormTransformerLayer(nn.Module):
    def __init__(self, dim, intermediate_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.linear1 = nn.Linear(dim, intermediate_dim)
        self.linear2 = nn.Linear(intermediate_dim, dim)
        self.act = nn.GELU()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Pre-norm: normalize before sub-layer
        normalized_x = self.norm1(x)
        attn_output = self.attention(normalized_x, normalized_x, normalized_x)[0]
        x = x + attn_output  # Residual connection

        normalized_x = self.norm2(x)
        mlp_output = self.linear1(normalized_x)
        mlp_output = self.act(mlp_output)
        mlp_output = self.linear2(mlp_output)
        x = x + mlp_output   # Residual connection
        return x
