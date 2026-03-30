# Transformer
## Chapter 19

---

### Padding

# 02 — Padding / 02 Padding

**Chapter 19 — File 1 of 9 / 第19章 — 第1个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Create mask which marks the zero padding values in the input by a 1**.

本脚本演示 **Create mask which marks the zero padding values in the input by a 1**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from numpy import array
from tensorflow import math, cast, float32

def padding_mask(input):
```

---
## Step 2 — Create mask which marks the zero padding values in the input by a 1

```python
mask = math.equal(input, 0)
    mask = cast(mask, float32)

    return mask

input = array([1, 2, 3, 4, 0, 0, 0])
print(padding_mask(input))
```

---
## Learning Notes / 学习笔记

- **概念**: Create mask which marks the zero padding values in the input by a 1 是机器学习中的常用技术。  
  *Create mask which marks the zero padding values in the input by a 1 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Padding / 02 Padding
# Complete Code / 完整代码
# ===============================

from numpy import array
from tensorflow import math, cast, float32

def padding_mask(input):
    # Create mask which marks the zero padding values in the input by a 1
    mask = math.equal(input, 0)
    mask = cast(mask, float32)

    return mask

input = array([1, 2, 3, 4, 0, 0, 0])
print(padding_mask(input))
```

---

➡️ **Next / 下一步**: File 2 of 9

---

### Lookahead

# 04 — Lookahead / 04 Lookahead

**Chapter 19 — File 2 of 9 / 第19章 — 第2个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Mask out future entries by marking them with a 1.0**.

本脚本演示 **Mask out future entries by marking them with a 1.0**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from tensorflow import linalg, ones

def lookahead_mask(shape):
```

---
## Step 2 — Mask out future entries by marking them with a 1.0

```python
mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0)

    return mask

print(lookahead_mask(5))
```

---
## Learning Notes / 学习笔记

- **概念**: Mask out future entries by marking them with a 1.0 是机器学习中的常用技术。  
  *Mask out future entries by marking them with a 1.0 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Lookahead / 04 Lookahead
# Complete Code / 完整代码
# ===============================

from tensorflow import linalg, ones

def lookahead_mask(shape):
    # Mask out future entries by marking them with a 1.0
    mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0)

    return mask

print(lookahead_mask(5))
```

---

➡️ **Next / 下一步**: File 3 of 9

---

### Transformer

# 09 — Transformer / 数据变换

**Chapter 19 — File 3 of 9 / 第19章 — 第3个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Set up the encoder**.

本脚本演示 **Set up the encoder**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
from encoder import Encoder
from decoder import Decoder
from tensorflow import math, cast, float32, linalg, ones, maximum, newaxis
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class TransformerModel(Model):
    def __init__(self, enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length,
                       h, d_k, d_v, d_model, d_ff_inner, n, rate, **kwargs):
        super().__init__(**kwargs)
```

---
## Step 2 — Set up the encoder

```python
self.encoder = Encoder(enc_vocab_size, enc_seq_length, h, d_k, d_v,
                               d_model, d_ff_inner, n, rate)
```

---
## Step 3 — Set up the decoder

```python
self.decoder = Decoder(dec_vocab_size, dec_seq_length, h, d_k, d_v,
                               d_model, d_ff_inner, n, rate)
```

---
## Step 4 — Define the final dense layer

```python
self.model_last_layer = Dense(dec_vocab_size)

    def padding_mask(self, input):
```

---
## Step 5 — Create mask which marks the zero padding values in the input by a 1.0

```python
mask = math.equal(input, 0)
        mask = cast(mask, float32)
```

---
## Step 6 — The shape of the mask should be broadcastable to the shape
of the attention weights that it will be masking later on

```python
return mask[:, newaxis, newaxis, :]

    def lookahead_mask(self, shape):
```

---
## Step 7 — Mask out future entries by marking them with a 1.0

```python
mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0)

        return mask

    def call(self, encoder_input, decoder_input, training):
```

---
## Step 8 — Create padding mask to mask the encoder inputs and the encoder
outputs in the decoder

```python
enc_padding_mask = self.padding_mask(encoder_input)
```

---
## Step 9 — Create and combine padding and look-ahead masks to be fed into the decoder

```python
dec_in_padding_mask = self.padding_mask(decoder_input)
        dec_in_lookahead_mask = self.lookahead_mask(decoder_input.shape[1])
        dec_in_lookahead_mask = maximum(dec_in_padding_mask, dec_in_lookahead_mask)
```

---
## Step 10 — Feed the input into the encoder

```python
encoder_output = self.encoder(encoder_input, enc_padding_mask, training)
```

---
## Step 11 — Feed the encoder output into the decoder

```python
decoder_output = self.decoder(decoder_input, encoder_output,
                                      dec_in_lookahead_mask, enc_padding_mask, training)
```

---
## Step 12 — Pass the decoder output through a final dense layer

```python
model_output = self.model_last_layer(decoder_output)

        return model_output
```

---
## Learning Notes / 学习笔记

- **概念**: Set up the encoder 是机器学习中的常用技术。  
  *Set up the encoder is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Transformer / 数据变换
# Complete Code / 完整代码
# ===============================

from encoder import Encoder
from decoder import Decoder
from tensorflow import math, cast, float32, linalg, ones, maximum, newaxis
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class TransformerModel(Model):
    def __init__(self, enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length,
                       h, d_k, d_v, d_model, d_ff_inner, n, rate, **kwargs):
        super().__init__(**kwargs)

        # Set up the encoder
        self.encoder = Encoder(enc_vocab_size, enc_seq_length, h, d_k, d_v,
                               d_model, d_ff_inner, n, rate)

        # Set up the decoder
        self.decoder = Decoder(dec_vocab_size, dec_seq_length, h, d_k, d_v,
                               d_model, d_ff_inner, n, rate)

        # Define the final dense layer
        self.model_last_layer = Dense(dec_vocab_size)

    def padding_mask(self, input):
        # Create mask which marks the zero padding values in the input by a 1.0
        mask = math.equal(input, 0)
        mask = cast(mask, float32)

        # The shape of the mask should be broadcastable to the shape
        # of the attention weights that it will be masking later on
        return mask[:, newaxis, newaxis, :]

    def lookahead_mask(self, shape):
        # Mask out future entries by marking them with a 1.0
        mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0)

        return mask

    def call(self, encoder_input, decoder_input, training):

        # Create padding mask to mask the encoder inputs and the encoder
        # outputs in the decoder
        enc_padding_mask = self.padding_mask(encoder_input)

        # Create and combine padding and look-ahead masks to be fed into the decoder
        dec_in_padding_mask = self.padding_mask(decoder_input)
        dec_in_lookahead_mask = self.lookahead_mask(decoder_input.shape[1])
        dec_in_lookahead_mask = maximum(dec_in_padding_mask, dec_in_lookahead_mask)

        # Feed the input into the encoder
        encoder_output = self.encoder(encoder_input, enc_padding_mask, training)

        # Feed the encoder output into the decoder
        decoder_output = self.decoder(decoder_input, encoder_output,
                                      dec_in_lookahead_mask, enc_padding_mask, training)

        # Pass the decoder output through a final dense layer
        model_output = self.model_last_layer(decoder_output)

        return model_output
```

---

➡️ **Next / 下一步**: File 4 of 9

---

### Create Model

# 13 — Create Model / 13 Create Model

**Chapter 19 — File 4 of 9 / 第19章 — 第4个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Create model**.

本脚本演示 **Create model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
from model import TransformerModel

enc_vocab_size = 20 # Vocabulary size for the encoder
dec_vocab_size = 20 # Vocabulary size for the decoder

enc_seq_length = 5  # Maximum length of the input sequence
dec_seq_length = 5  # Maximum length of the target sequence

h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_ff = 2048  # Dimensionality of the inner fully connected layer
d_model = 512  # Dimensionality of the model sub-layers' outputs
n = 6  # Number of layers in the encoder stack

dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers
```

---
## Step 2 — Create model

```python
training_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length,
                                  dec_seq_length, h, d_k, d_v, d_model, d_ff, n,
                                  dropout_rate)
```

---
## Learning Notes / 学习笔记

- **概念**: Create model 是机器学习中的常用技术。  
  *Create model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Create Model / 13 Create Model
# Complete Code / 完整代码
# ===============================

from model import TransformerModel

enc_vocab_size = 20 # Vocabulary size for the encoder
dec_vocab_size = 20 # Vocabulary size for the decoder

enc_seq_length = 5  # Maximum length of the input sequence
dec_seq_length = 5  # Maximum length of the target sequence

h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_ff = 2048  # Dimensionality of the inner fully connected layer
d_model = 512  # Dimensionality of the model sub-layers' outputs
n = 6  # Number of layers in the encoder stack

dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers

# Create model
training_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length,
                                  dec_seq_length, h, d_k, d_v, d_model, d_ff, n,
                                  dropout_rate)
```

---

➡️ **Next / 下一步**: File 5 of 9

---

### Chapter Summary

# Chapter 19 Summary / 第19章总结

## Theme / 主题: Chapter 19 / Chapter 19

This chapter contains **9 code files** demonstrating chapter 19.

本章包含 **9 个代码文件**，演示Chapter 19。

---
## Evolution / 演化路线

  1. `02_padding.ipynb` — Padding
  2. `04_lookahead.ipynb` — Lookahead
  3. `09_transformer.ipynb` — Transformer
  4. `13_create_model.ipynb` — Create Model
  5. `decoder.ipynb` — Decoder
  6. `encoder.ipynb` — Encoder
  7. `model.ipynb` — Model
  8. `multihead_attention.ipynb` — Multihead Attention
  9. `positional_encoding.ipynb` — Positional Encoding

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 19) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 19）是机器学习流水线中的基础构建块。

---

### Model

# 01 — Model / Model

**Chapter 19 — File 7 of 9 / 第19章 — 第7个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Set up the encoder**.

本脚本演示 **Set up the encoder**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
from encoder import Encoder
from decoder import Decoder
from tensorflow import math, cast, float32, linalg, ones, maximum, newaxis
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class TransformerModel(Model):
    def __init__(self, enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length,
                       h, d_k, d_v, d_model, d_ff_inner, n, rate, **kwargs):
        super().__init__(**kwargs)
```

---
## Step 2 — Set up the encoder

```python
self.encoder = Encoder(enc_vocab_size, enc_seq_length, h, d_k, d_v,
                               d_model, d_ff_inner, n, rate)
```

---
## Step 3 — Set up the decoder

```python
self.decoder = Decoder(dec_vocab_size, dec_seq_length, h, d_k, d_v,
                               d_model, d_ff_inner, n, rate)
```

---
## Step 4 — Define the final dense layer

```python
self.model_last_layer = Dense(dec_vocab_size)

    def padding_mask(self, input):
```

---
## Step 5 — Create mask which marks the zero padding values in the input by a 1.0

```python
mask = math.equal(input, 0)
        mask = cast(mask, float32)
```

---
## Step 6 — The shape of the mask should be broadcastable to the shape
of the attention weights that it will be masking later on

```python
return mask[:, newaxis, newaxis, :]

    def lookahead_mask(self, shape):
```

---
## Step 7 — Mask out future entries by marking them with a 1.0

```python
mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0)

        return mask

    def call(self, encoder_input, decoder_input, training):
```

---
## Step 8 — Create padding mask to mask the encoder inputs and the encoder
outputs in the decoder

```python
enc_padding_mask = self.padding_mask(encoder_input)
```

---
## Step 9 — Create and combine padding and look-ahead masks to be fed into the decoder

```python
dec_in_padding_mask = self.padding_mask(decoder_input)
        dec_in_lookahead_mask = self.lookahead_mask(decoder_input.shape[1])
        dec_in_lookahead_mask = maximum(dec_in_padding_mask, dec_in_lookahead_mask)
```

---
## Step 10 — Feed the input into the encoder

```python
encoder_output = self.encoder(encoder_input, enc_padding_mask, training)
```

---
## Step 11 — Feed the encoder output into the decoder

```python
decoder_output = self.decoder(decoder_input, encoder_output,
                                      dec_in_lookahead_mask, enc_padding_mask, training)
```

---
## Step 12 — Pass the decoder output through a final dense layer

```python
model_output = self.model_last_layer(decoder_output)

        return model_output
```

---
## Learning Notes / 学习笔记

- **概念**: Set up the encoder 是机器学习中的常用技术。  
  *Set up the encoder is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Model / Model
# Complete Code / 完整代码
# ===============================

from encoder import Encoder
from decoder import Decoder
from tensorflow import math, cast, float32, linalg, ones, maximum, newaxis
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class TransformerModel(Model):
    def __init__(self, enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length,
                       h, d_k, d_v, d_model, d_ff_inner, n, rate, **kwargs):
        super().__init__(**kwargs)

        # Set up the encoder
        self.encoder = Encoder(enc_vocab_size, enc_seq_length, h, d_k, d_v,
                               d_model, d_ff_inner, n, rate)

        # Set up the decoder
        self.decoder = Decoder(dec_vocab_size, dec_seq_length, h, d_k, d_v,
                               d_model, d_ff_inner, n, rate)

        # Define the final dense layer
        self.model_last_layer = Dense(dec_vocab_size)

    def padding_mask(self, input):
        # Create mask which marks the zero padding values in the input by a 1.0
        mask = math.equal(input, 0)
        mask = cast(mask, float32)

        # The shape of the mask should be broadcastable to the shape
        # of the attention weights that it will be masking later on
        return mask[:, newaxis, newaxis, :]

    def lookahead_mask(self, shape):
        # Mask out future entries by marking them with a 1.0
        mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0)

        return mask

    def call(self, encoder_input, decoder_input, training):

        # Create padding mask to mask the encoder inputs and the encoder
        # outputs in the decoder
        enc_padding_mask = self.padding_mask(encoder_input)

        # Create and combine padding and look-ahead masks to be fed into the decoder
        dec_in_padding_mask = self.padding_mask(decoder_input)
        dec_in_lookahead_mask = self.lookahead_mask(decoder_input.shape[1])
        dec_in_lookahead_mask = maximum(dec_in_padding_mask, dec_in_lookahead_mask)

        # Feed the input into the encoder
        encoder_output = self.encoder(encoder_input, enc_padding_mask, training)

        # Feed the encoder output into the decoder
        decoder_output = self.decoder(decoder_input, encoder_output,
                                      dec_in_lookahead_mask, enc_padding_mask, training)

        # Pass the decoder output through a final dense layer
        model_output = self.model_last_layer(decoder_output)

        return model_output
```

---

➡️ **Next / 下一步**: File 8 of 9

---

### Multihead Attention

# 01 — Multihead Attention / 注意力机制

**Chapter 19 — File 8 of 9 / 第19章 — 第8个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Implementing the Scaled-Dot Product Attention**.

本脚本演示 **Implementing the Scaled-Dot Product Attention**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


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

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |

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

➡️ **Next / 下一步**: File 9 of 9

---

### Positional Encoding

# 01 — Positional Encoding / Positional Encoding

**Chapter 19 — File 9 of 9 / 第19章 — 第9个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Positional Encoding**.

本脚本演示 **Positional Encoding**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


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

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `np.zeros` | 全零数组 | Array filled with zeros |
| `numpy` | 数值计算库 | Numerical computing library |

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
