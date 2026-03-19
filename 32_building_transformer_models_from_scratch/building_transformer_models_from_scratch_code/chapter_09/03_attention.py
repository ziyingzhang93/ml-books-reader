import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_prob=0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_prob = dropout_prob

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_length = x.size(1)

        # Project queries, keys, and values
        q = self.q_proj(x) \
                .view(batch_size, seq_length, self.num_heads, self.head_dim) \
                .transpose(1, 2)
        k = self.k_proj(x) \
                .view(batch_size, seq_length, self.num_heads, self.head_dim) \
                .transpose(1, 2)
        v = self.v_proj(x) \
                .view(batch_size, seq_length, self.num_heads, self.head_dim) \
                .transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply mask to attention scores
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Apply softmax to compute the attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout
        if self.dropout_prob:
            attn_weights = F.dropout(attn_weights, p=self.dropout_prob)

        # Apply attention weights to values
        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, self.d_model)

        return self.out_proj(context)
