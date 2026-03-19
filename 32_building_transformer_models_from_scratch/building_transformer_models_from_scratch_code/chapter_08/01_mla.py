import math
import torch
import torch.nn as nn

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model=128*128, num_heads=128, q_latent_dim=12, kv_latent_dim=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_latent_dim = q_latent_dim
        self.kv_latent_dim = kv_latent_dim
        head_dim = d_model // num_heads

        # Query projections
        self.Wq_d = nn.Linear(d_model, q_latent_dim)

        # Precomputed matrix multiplications of W_q^U and W_k^U, for multiple heads
        self.W_qk = nn.Linear(q_latent_dim, num_heads * kv_latent_dim)

        # Key/Value latent projections
        self.Wkv_d = nn.Linear(d_model, kv_latent_dim)
        self.Wv_u = nn.Linear(kv_latent_dim, num_heads * head_dim)

        # Output projection
        self.Wo = nn.Linear(num_heads * head_dim, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        # Projections of input into latent spaces
        C_q = self.Wq_d(x)     # shape: (batch_size, seq_len, q_latent_dim)
        C_kv = self.Wkv_d(x)   # shape: (batch_size, seq_len, kv_latent_dim)

        # Attention score, shape: (batch_size, num_heads, seq_len, seq_len)
        C_qW_qk = self.W_qk(C_q) \
                      .view(batch_size, seq_len, self.num_heads, self.kv_latent_dim)
        scores = torch.matmul(C_qW_qk.transpose(1, 2),
                              C_kv.transpose(-2, -1)[:, None, ...]
                 ) / math.sqrt(self.kv_latent_dim)

        # Attention computation
        attn_weight = torch.softmax(scores, dim=-1)
        # Restore V from latent space
        V = self.Wv_u(C_kv).view(batch_size, seq_len, self.num_heads, -1)
        # Compute attention output, shape: (batch_size, seq_len, num_heads, head_dim)
        output = torch.matmul(attn_weight, V.transpose(1,2)).transpose(1,2).contiguous()
        # Concatentate the heads, then apply output projection
        output = self.Wo(output.view(batch_size, seq_len, -1))
        return output
