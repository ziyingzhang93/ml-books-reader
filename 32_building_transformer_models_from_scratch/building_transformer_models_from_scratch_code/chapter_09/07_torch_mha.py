import torch

dim = 16
num_heads = 4
attn_layer = torch.nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)

# Input tensor: 0 = padding
batch = torch.tensor([
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 0, 0, 0],
    [1, 2, 3, 4, 0, 0]
])
batch_size, seq_len = batch.shape
x = torch.randn(batch_size, seq_len, dim)

padding_mask = (batch == 0)
y = attn_layer(x, x, x, key_padding_mask=padding_mask, attn_mask=None)
