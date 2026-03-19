import torch.nn as nn
import torch.nn.functional as F

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

        # Compute attention scores using PyTorch's built-in function
        attn_output = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

        # Output projection
        context = attn_output.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, self.d_model)
        return self.out_proj(context)
