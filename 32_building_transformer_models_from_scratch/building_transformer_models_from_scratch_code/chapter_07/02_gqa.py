import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads     # num of query heads
        self.num_groups = num_groups
        self.group_size = num_heads // num_groups
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, self.num_groups * self.head_dim)
        self.v_proj = nn.Linear(d_model, self.num_groups * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)

        # Project queries, keys, and values
        q = self.q_proj(x) \
                .view(batch_size, seq_length, self.num_heads, self.head_dim) \
                .transpose(1, 2)
        k = self.k_proj(x) \
                .view(batch_size, seq_length, self.num_groups, self.head_dim) \
                .transpose(1, 2)
        v = self.v_proj(x) \
                .view(batch_size, seq_length, self.num_groups, self.head_dim) \
                .transpose(1, 2)

        # Expand k and v to match the number of query heads
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        # optional: attn_weights = F.dropout(attn_weights, p=0.2)

        # Apply attention weights to values
        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, self.d_model)

        return self.out_proj(context)
