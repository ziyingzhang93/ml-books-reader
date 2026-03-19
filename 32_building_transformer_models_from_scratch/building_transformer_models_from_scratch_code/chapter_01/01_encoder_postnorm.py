import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ff_proj = nn.Linear(d_model, d_ff)
        self.output_proj = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.ReLU()

    def forward(self, x):
        """Process the input sequence x

        Args:
            x: The input sequence of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The processed sequence of shape (batch_size, seq_len, d_model).
        """
        # Self-attention sublayer
        residual = x
        x = self.attention(x, x, x)
        x = self.norm1(x[0] + residual)

        # Feed-forward sublayer
        residual = x
        x = self.act(self.ff_proj(x))
        x = self.act(self.output_proj(x))
        x = self.norm2(x + residual)

        return x

seq = torch.randn(3, 7, 16)
layer = TransformerEncoderLayer(16, 32, 4)
out_seq = layer(seq)
print({name: weight.shape for name, weight in layer.state_dict().items()})
print(out_seq.shape)
