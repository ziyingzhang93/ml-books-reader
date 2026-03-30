# 注意力与Transformer / Transformer Models with Attention
## Chapter 16

---

### Testattention

# 11 — Testattention / 注意力机制

**Chapter 16 — File 1 of 1 / 第16章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **Implementing the Scaled-Dot Product Attention**.

本脚本演示 **Implementing the Scaled-Dot Product Attention**。

---
## Step 1 — Step 1

```python
from numpy import random
from tensorflow import math, matmul, reshape, shape, transpose, cast, float32
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.backend import softmax
```

---
## Step 2 — Implementing the Scaled-Dot Product Attention

```python
class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, queries, keys, values, d_k, mask=None):
```

---
## Step 3 — Scoring the queries against the keys after transposing the latter, and scaling

```python
scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))
```

---
## Step 4 — Apply mask to the attention scores

```python
if mask is not None:
            scores += -1e9 * mask
```

---
## Step 5 — Computing the weights by a softmax operation

```python
weights = softmax(scores)
```

---
## Step 6 — Computing the attention by a weighted sum of the value vectors

```python
return matmul(weights, values)
```

---
## Step 7 — Implementing the Multi-Head Attention

```python
class MultiHeadAttention(Layer):
    def __init__(self, h, d_k, d_v, d_model, **kwargs):
        super().__init__(**kwargs)
        self.attention = DotProductAttention()  # Scaled dot product attention
        self.heads = h  # Number of attention heads to use
        self.d_k = d_k  # Dimensionality of the linearly projected queries and keys
        self.d_v = d_v  # Dimensionality of the linearly projected values
        self.d_model = d_model  # Dimensionality of the model
        self.W_q = Dense(d_k)   # Learned projection matrix for the queries
        self.W_k = Dense(d_k)   # Learned projection matrix for the keys
        self.W_v = Dense(d_v)   # Learned projection matrix for the values
        self.W_o = Dense(d_model) # Learned projection matrix for the multi-head output

    def reshape_tensor(self, x, heads, flag):
        if flag:
```

---
## Step 8 — Tensor shape after reshaping and transposing:
(batch_size, heads, seq_length, -1)

```python
x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
```

---
## Step 9 — Reverting the reshaping and transposing operations:
(batch_size, seq_length, d_k)

```python
x = transpose(x, perm=(0, 2, 1, 3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.d_k))
        return x

    def call(self, queries, keys, values, mask=None):
```

---
## Step 10 — Rearrange the queries to be able to compute all heads in parallel

```python
q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
```

---
## Step 11 — Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
Rearrange the keys to be able to compute all heads in parallel

```python
k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
```

---
## Step 12 — Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
Rearrange the values to be able to compute all heads in parallel

```python
v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
```

---
## Step 13 — Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
Compute the multi-head attention output using the reshaped queries,
keys, and values

```python
o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
```

---
## Step 14 — Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
Rearrange back the output into concatenated form

```python
output = self.reshape_tensor(o_reshaped, self.heads, False)
```

---
## Step 15 — Resulting tensor shape: (batch_size, input_seq_length, d_v)
Apply one final linear projection to the output to generate the multi-head
attention. Resulting tensor shape: (batch_size, input_seq_length, d_model)

```python
return self.W_o(output)

input_seq_length = 5  # Maximum length of the input sequence
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of the model sub-layers' outputs
batch_size = 64  # Batch size from the training process

queries = random.random((batch_size, input_seq_length, d_k))
keys = random.random((batch_size, input_seq_length, d_k))
values = random.random((batch_size, input_seq_length, d_v))

multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
print(multihead_attention(queries, keys, values))
```

---
## Learning Notes / 学习笔记

- **概念**: Implementing the Scaled-Dot Product Attention 是机器学习中的常用技术。  
  *Implementing the Scaled-Dot Product Attention is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Testattention / 注意力机制
# Complete Code / 完整代码
# ===============================

from numpy import random
from tensorflow import math, matmul, reshape, shape, transpose, cast, float32
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.backend import softmax

# Implementing the Scaled-Dot Product Attention
class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, queries, keys, values, d_k, mask=None):
        # Scoring the queries against the keys after transposing the latter, and scaling
        scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))

        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask

        # Computing the weights by a softmax operation
        weights = softmax(scores)

        # Computing the attention by a weighted sum of the value vectors
        return matmul(weights, values)

# Implementing the Multi-Head Attention
class MultiHeadAttention(Layer):
    def __init__(self, h, d_k, d_v, d_model, **kwargs):
        super().__init__(**kwargs)
        self.attention = DotProductAttention()  # Scaled dot product attention
        self.heads = h  # Number of attention heads to use
        self.d_k = d_k  # Dimensionality of the linearly projected queries and keys
        self.d_v = d_v  # Dimensionality of the linearly projected values
        self.d_model = d_model  # Dimensionality of the model
        self.W_q = Dense(d_k)   # Learned projection matrix for the queries
        self.W_k = Dense(d_k)   # Learned projection matrix for the keys
        self.W_v = Dense(d_v)   # Learned projection matrix for the values
        self.W_o = Dense(d_model) # Learned projection matrix for the multi-head output

    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing:
            # (batch_size, heads, seq_length, -1)
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations:
            # (batch_size, seq_length, d_k)
            x = transpose(x, perm=(0, 2, 1, 3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.d_k))
        return x

    def call(self, queries, keys, values, mask=None):
        # Rearrange the queries to be able to compute all heads in parallel
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the keys to be able to compute all heads in parallel
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the values to be able to compute all heads in parallel
        v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Compute the multi-head attention output using the reshaped queries,
        # keys, and values
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange back the output into concatenated form
        output = self.reshape_tensor(o_reshaped, self.heads, False)
        # Resulting tensor shape: (batch_size, input_seq_length, d_v)

        # Apply one final linear projection to the output to generate the multi-head
        # attention. Resulting tensor shape: (batch_size, input_seq_length, d_model)
        return self.W_o(output)

input_seq_length = 5  # Maximum length of the input sequence
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of the model sub-layers' outputs
batch_size = 64  # Batch size from the training process

queries = random.random((batch_size, input_seq_length, d_k))
keys = random.random((batch_size, input_seq_length, d_k))
values = random.random((batch_size, input_seq_length, d_v))

multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
print(multihead_attention(queries, keys, values))
```

---

### Chapter Summary / 章节总结

# Chapter 16 Summary / 第16章总结

## Theme / 主题: Chapter 16 / Chapter 16

This chapter contains **1 code files** demonstrating chapter 16.

本章包含 **1 个代码文件**，演示Chapter 16。

---
## Evolution / 演化路线

  1. `11_testattention.ipynb` — Testattention

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 16) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 16）是机器学习流水线中的基础构建块。

---
