# 注意力与Transformer / Transformer Models with Attention
## Chapter 17

---

### Encoder

# 09 — Encoder / 数据编码

**Chapter 17 — File 1 of 4 / 第17章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Implementing the Add & Norm Layer**.

本脚本演示 **Implementing the Add & Norm Layer**。

---
## Step 1 — Step 1

```python
from tensorflow.keras.layers import LayerNormalization, Layer, Dense, ReLU, Dropout
from multihead_attention import MultiHeadAttention
from positional_encoding import PositionEmbeddingFixedWeights
```

---
## Step 2 — Implementing the Add & Norm Layer

```python
class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = LayerNormalization()  # Layer normalization layer

    def call(self, x, sublayer_x):
```

---
## Step 3 — The sublayer input and output need to be of the same shape to be summed

```python
add = x + sublayer_x
```

---
## Step 4 — Apply layer normalization to the sum

```python
return self.layer_norm(add)
```

---
## Step 5 — Implementing the Feed-Forward Layer

```python
class FeedForward(Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        super().__init__(**kwargs)
        self.fully_connected1 = Dense(d_ff)  # First fully connected layer
        self.fully_connected2 = Dense(d_model)  # Second fully connected layer
        self.activation = ReLU()  # ReLU activation layer

    def call(self, x):
```

---
## Step 6 — The input is passed into the two fully-connected layers, with a ReLU in between

```python
x_fc1 = self.fully_connected1(x)

        return self.fully_connected2(self.activation(x_fc1))
```

---
## Step 7 — Implementing the Encoder Layer

```python
class EncoderLayer(Layer):
    def __init__(self, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super().__init__(**kwargs)
        self.multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()

    def call(self, x, padding_mask, training):
```

---
## Step 8 — Multi-head attention layer

```python
multihead_output = self.multihead_attention(x, x, x, padding_mask)
```

---
## Step 9 — Expected output shape = (batch_size, sequence_length, d_model)
Add in a dropout layer

```python
multihead_output = self.dropout1(multihead_output, training=training)
```

---
## Step 10 — Followed by an Add & Norm layer

```python
addnorm_output = self.add_norm1(x, multihead_output)
```

---
## Step 11 — Expected output shape = (batch_size, sequence_length, d_model)
Followed by a fully connected layer

```python
feedforward_output = self.feed_forward(addnorm_output)
```

---
## Step 12 — Expected output shape = (batch_size, sequence_length, d_model)
Add in another dropout layer

```python
feedforward_output = self.dropout2(feedforward_output, training=training)
```

---
## Step 13 — Followed by another Add & Norm layer

```python
return self.add_norm2(addnorm_output, feedforward_output)
```

---
## Step 14 — Implementing the Encoder

```python
class Encoder(Layer):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate,
                       **kwargs):
        super().__init__(**kwargs)
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size,
                                                          d_model)
        self.dropout = Dropout(rate)
        self.encoder_layer = [EncoderLayer(h, d_k, d_v, d_model, d_ff, rate)
                              for _ in range(n)]

    def call(self, input_sentence, padding_mask, training):
```

---
## Step 15 — Generate the positional encoding

```python
pos_encoding_output = self.pos_encoding(input_sentence)
```

---
## Step 16 — Expected output shape = (batch_size, sequence_length, d_model)
Add in a dropout layer

```python
x = self.dropout(pos_encoding_output, training=training)
```

---
## Step 17 — Pass on the positional encoded values to each encoder layer

```python
for i, layer in enumerate(self.encoder_layer):
            x = layer(x, padding_mask, training)

        return x
```

---
## Learning Notes / 学习笔记

- **概念**: Implementing the Add & Norm Layer 是机器学习中的常用技术。  
  *Implementing the Add & Norm Layer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Encoder / 数据编码
# Complete Code / 完整代码
# ===============================

from tensorflow.keras.layers import LayerNormalization, Layer, Dense, ReLU, Dropout
from multihead_attention import MultiHeadAttention
from positional_encoding import PositionEmbeddingFixedWeights

# Implementing the Add & Norm Layer
class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = LayerNormalization()  # Layer normalization layer

    def call(self, x, sublayer_x):
        # The sublayer input and output need to be of the same shape to be summed
        add = x + sublayer_x

        # Apply layer normalization to the sum
        return self.layer_norm(add)

# Implementing the Feed-Forward Layer
class FeedForward(Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        super().__init__(**kwargs)
        self.fully_connected1 = Dense(d_ff)  # First fully connected layer
        self.fully_connected2 = Dense(d_model)  # Second fully connected layer
        self.activation = ReLU()  # ReLU activation layer

    def call(self, x):
        # The input is passed into the two fully-connected layers, with a ReLU in between
        x_fc1 = self.fully_connected1(x)

        return self.fully_connected2(self.activation(x_fc1))

# Implementing the Encoder Layer
class EncoderLayer(Layer):
    def __init__(self, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super().__init__(**kwargs)
        self.multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()

    def call(self, x, padding_mask, training):
        # Multi-head attention layer
        multihead_output = self.multihead_attention(x, x, x, padding_mask)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        multihead_output = self.dropout1(multihead_output, training=training)

        # Followed by an Add & Norm layer
        addnorm_output = self.add_norm1(x, multihead_output)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in another dropout layer
        feedforward_output = self.dropout2(feedforward_output, training=training)

        # Followed by another Add & Norm layer
        return self.add_norm2(addnorm_output, feedforward_output)

# Implementing the Encoder
class Encoder(Layer):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate,
                       **kwargs):
        super().__init__(**kwargs)
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size,
                                                          d_model)
        self.dropout = Dropout(rate)
        self.encoder_layer = [EncoderLayer(h, d_k, d_v, d_model, d_ff, rate)
                              for _ in range(n)]

    def call(self, input_sentence, padding_mask, training):
        # Generate the positional encoding
        pos_encoding_output = self.pos_encoding(input_sentence)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        x = self.dropout(pos_encoding_output, training=training)

        # Pass on the positional encoded values to each encoder layer
        for i, layer in enumerate(self.encoder_layer):
            x = layer(x, padding_mask, training)

        return x
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Testencoder

# 13 — Testencoder / 数据编码

**Chapter 17 — File 2 of 4 / 第17章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Implementing the Add & Norm Layer**.

本脚本演示 **Implementing the Add & Norm Layer**。

---
## Step 1 — Step 1

```python
import numpy as np
from numpy import random
from tensorflow.keras.layers import LayerNormalization, Layer, Dense, ReLU, Dropout
from multihead_attention import MultiHeadAttention
from positional_encoding import PositionEmbeddingFixedWeights
```

---
## Step 2 — Implementing the Add & Norm Layer

```python
class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = LayerNormalization()  # Layer normalization layer

    def call(self, x, sublayer_x):
```

---
## Step 3 — The sublayer input and output need to be of the same shape to be summed

```python
add = x + sublayer_x
```

---
## Step 4 — Apply layer normalization to the sum

```python
return self.layer_norm(add)
```

---
## Step 5 — Implementing the Feed-Forward Layer

```python
class FeedForward(Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        super().__init__(**kwargs)
        self.fully_connected1 = Dense(d_ff)  # First fully connected layer
        self.fully_connected2 = Dense(d_model)  # Second fully connected layer
        self.activation = ReLU()  # ReLU activation layer

    def call(self, x):
```

---
## Step 6 — The input is passed into the two fully-connected layers, with a ReLU in between

```python
x_fc1 = self.fully_connected1(x)

        return self.fully_connected2(self.activation(x_fc1))
```

---
## Step 7 — Implementing the Encoder Layer

```python
class EncoderLayer(Layer):
    def __init__(self, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super().__init__(**kwargs)
        self.multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()

    def call(self, x, padding_mask, training):
```

---
## Step 8 — Multi-head attention layer

```python
multihead_output = self.multihead_attention(x, x, x, padding_mask)
```

---
## Step 9 — Expected output shape = (batch_size, sequence_length, d_model)
Add in a dropout layer

```python
multihead_output = self.dropout1(multihead_output, training=training)
```

---
## Step 10 — Followed by an Add & Norm layer

```python
addnorm_output = self.add_norm1(x, multihead_output)
```

---
## Step 11 — Expected output shape = (batch_size, sequence_length, d_model)
Followed by a fully connected layer

```python
feedforward_output = self.feed_forward(addnorm_output)
```

---
## Step 12 — Expected output shape = (batch_size, sequence_length, d_model)
Add in another dropout layer

```python
feedforward_output = self.dropout2(feedforward_output, training=training)
```

---
## Step 13 — Followed by another Add & Norm layer

```python
return self.add_norm2(addnorm_output, feedforward_output)
```

---
## Step 14 — Implementing the Encoder

```python
class Encoder(Layer):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate,
                       **kwargs):
        super().__init__(**kwargs)
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size,
                                                          d_model)
        self.dropout = Dropout(rate)
        self.encoder_layer = [EncoderLayer(h, d_k, d_v, d_model, d_ff, rate)
                              for _ in range(n)]

    def call(self, input_sentence, padding_mask, training):
```

---
## Step 15 — Generate the positional encoding

```python
pos_encoding_output = self.pos_encoding(input_sentence)
```

---
## Step 16 — Expected output shape = (batch_size, sequence_length, d_model)
Add in a dropout layer

```python
x = self.dropout(pos_encoding_output, training=training)
```

---
## Step 17 — Pass on the positional encoded values to each encoder layer

```python
for i, layer in enumerate(self.encoder_layer):
            x = layer(x, padding_mask, training)

        return x

enc_vocab_size = 20 # Vocabulary size for the encoder
input_seq_length = 5  # Maximum length of the input sequence
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_ff = 2048  # Dimensionality of the inner fully connected layer
d_model = 512  # Dimensionality of the model sub-layers' outputs
n = 6  # Number of layers in the encoder stack

batch_size = 64  # Batch size from the training process
dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers

input_seq = random.random((batch_size, input_seq_length))

encoder = Encoder(enc_vocab_size, input_seq_length, h, d_k, d_v, d_model, d_ff, n,
                  dropout_rate)
print(encoder(input_seq, None, True))
```

---
## Learning Notes / 学习笔记

- **概念**: Implementing the Add & Norm Layer 是机器学习中的常用技术。  
  *Implementing the Add & Norm Layer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Testencoder / 数据编码
# Complete Code / 完整代码
# ===============================

import numpy as np
from numpy import random
from tensorflow.keras.layers import LayerNormalization, Layer, Dense, ReLU, Dropout
from multihead_attention import MultiHeadAttention
from positional_encoding import PositionEmbeddingFixedWeights

# Implementing the Add & Norm Layer
class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = LayerNormalization()  # Layer normalization layer

    def call(self, x, sublayer_x):
        # The sublayer input and output need to be of the same shape to be summed
        add = x + sublayer_x

        # Apply layer normalization to the sum
        return self.layer_norm(add)

# Implementing the Feed-Forward Layer
class FeedForward(Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        super().__init__(**kwargs)
        self.fully_connected1 = Dense(d_ff)  # First fully connected layer
        self.fully_connected2 = Dense(d_model)  # Second fully connected layer
        self.activation = ReLU()  # ReLU activation layer

    def call(self, x):
        # The input is passed into the two fully-connected layers, with a ReLU in between
        x_fc1 = self.fully_connected1(x)

        return self.fully_connected2(self.activation(x_fc1))

# Implementing the Encoder Layer
class EncoderLayer(Layer):
    def __init__(self, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super().__init__(**kwargs)
        self.multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()

    def call(self, x, padding_mask, training):
        # Multi-head attention layer
        multihead_output = self.multihead_attention(x, x, x, padding_mask)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        multihead_output = self.dropout1(multihead_output, training=training)

        # Followed by an Add & Norm layer
        addnorm_output = self.add_norm1(x, multihead_output)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in another dropout layer
        feedforward_output = self.dropout2(feedforward_output, training=training)

        # Followed by another Add & Norm layer
        return self.add_norm2(addnorm_output, feedforward_output)

# Implementing the Encoder
class Encoder(Layer):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate,
                       **kwargs):
        super().__init__(**kwargs)
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size,
                                                          d_model)
        self.dropout = Dropout(rate)
        self.encoder_layer = [EncoderLayer(h, d_k, d_v, d_model, d_ff, rate)
                              for _ in range(n)]

    def call(self, input_sentence, padding_mask, training):
        # Generate the positional encoding
        pos_encoding_output = self.pos_encoding(input_sentence)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        x = self.dropout(pos_encoding_output, training=training)

        # Pass on the positional encoded values to each encoder layer
        for i, layer in enumerate(self.encoder_layer):
            x = layer(x, padding_mask, training)

        return x

enc_vocab_size = 20 # Vocabulary size for the encoder
input_seq_length = 5  # Maximum length of the input sequence
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_ff = 2048  # Dimensionality of the inner fully connected layer
d_model = 512  # Dimensionality of the model sub-layers' outputs
n = 6  # Number of layers in the encoder stack

batch_size = 64  # Batch size from the training process
dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers

input_seq = random.random((batch_size, input_seq_length))

encoder = Encoder(enc_vocab_size, input_seq_length, h, d_k, d_v, d_model, d_ff, n,
                  dropout_rate)
print(encoder(input_seq, None, True))
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Chapter Summary / 章节总结

# Chapter 17 Summary / 第17章总结

## Theme / 主题: Chapter 17 / Chapter 17

This chapter contains **4 code files** demonstrating chapter 17.

本章包含 **4 个代码文件**，演示Chapter 17。

---
## Evolution / 演化路线

  1. `09_encoder.ipynb` — Encoder
  2. `13_testencoder.ipynb` — Testencoder
  3. `multihead_attention.ipynb` — Multihead Attention
  4. `positional_encoding.ipynb` — Positional Encoding

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 17) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 17）是机器学习流水线中的基础构建块。

---

### Multihead Attention

# 01 — Multihead Attention / 注意力机制

**Chapter 17 — File 3 of 4 / 第17章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Implementing the Scaled-Dot Product Attention**.

本脚本演示 **Implementing the Scaled-Dot Product Attention**。

---
## Step 1 — Step 1

```python
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
# Multihead Attention / 注意力机制
# Complete Code / 完整代码
# ===============================

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
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Positional Encoding

# 01 — Positional Encoding / Positional Encoding

**Chapter 17 — File 4 of 4 / 第17章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Positional Encoding**.

本脚本演示 **Positional Encoding**。

---
## Step 1 — Step 1

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Layer

class PositionEmbeddingFixedWeights(Layer):
    def __init__(self, seq_length, vocab_size, output_dim, **kwargs):
        super().__init__(**kwargs)
        word_embedding_matrix = self.get_position_encoding(vocab_size, output_dim)
        pos_embedding_matrix = self.get_position_encoding(seq_length, output_dim)
        self.word_embedding_layer = Embedding(
            input_dim=vocab_size, output_dim=output_dim,
            weights=[word_embedding_matrix],
            trainable=False
        )
        self.position_embedding_layer = Embedding(
            input_dim=seq_length, output_dim=output_dim,
            weights=[pos_embedding_matrix],
            trainable=False
        )

    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P


    def call(self, inputs):
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_words + embedded_indices
```

---
## Learning Notes / 学习笔记

- **概念**: Positional Encoding 是机器学习中的常用技术。  
  *Positional Encoding is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Positional Encoding / Positional Encoding
# Complete Code / 完整代码
# ===============================

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Layer

class PositionEmbeddingFixedWeights(Layer):
    def __init__(self, seq_length, vocab_size, output_dim, **kwargs):
        super().__init__(**kwargs)
        word_embedding_matrix = self.get_position_encoding(vocab_size, output_dim)
        pos_embedding_matrix = self.get_position_encoding(seq_length, output_dim)
        self.word_embedding_layer = Embedding(
            input_dim=vocab_size, output_dim=output_dim,
            weights=[word_embedding_matrix],
            trainable=False
        )
        self.position_embedding_layer = Embedding(
            input_dim=seq_length, output_dim=output_dim,
            weights=[pos_embedding_matrix],
            trainable=False
        )

    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P


    def call(self, inputs):
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_words + embedded_indices
```

---
