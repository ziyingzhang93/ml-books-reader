import torch.nn as nn

class BertLayer(nn.Module):
    def __init__(self, dim, intermediate_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.linear1 = nn.Linear(dim, intermediate_dim)
        self.linear2 = nn.Linear(intermediate_dim, dim)
        self.act = nn.GELU()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Skip connection around attention sub-layer
        attn_output = self.attention(x, x, x)[0]  # extract first element of the tuple
        x = x + attn_output  # Residual connection
        x = self.norm1(x)    # Layer normalization

        # Skip connection around MLP sub-layer
        mlp_output = self.linear1(x)
        mlp_output = self.act(mlp_output)
        mlp_output = self.linear2(mlp_output)
        x = x + mlp_output   # Residual connection
        x = self.norm2(x)    # Layer normalization
        return x
