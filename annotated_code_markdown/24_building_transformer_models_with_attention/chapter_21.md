# 注意力与Transformer / Transformer Models with Attention
## Chapter 21

---

### Prepare



---

### Model



---

### Plotting

# 03 — Plotting / 03 Plotting

**Chapter 21 — File 3 of 9 / 第21章 — 第3个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Load the training and validation loss dictionaries**.

本脚本演示 **Load the training and validation loss dictionaries**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  📈 可视化结果 / Visualize Results
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — Step 1

```python
from pickle import load
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib.pylab import plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
```

---
## Step 2 — Load the training and validation loss dictionaries

```python
train_loss = load(open('train_loss.pkl', 'rb'))
val_loss = load(open('val_loss.pkl', 'rb'))
```

---
## Step 3 — Retrieve each dictionary's values

```python
# 转换为NumPy数组 / Convert to NumPy array
train_values = train_loss.values()
# 转换为NumPy数组 / Convert to NumPy array
val_values = val_loss.values()
```

---
## Step 4 — Generate a sequence of integers to represent the epoch numbers

```python
# 生成整数序列 / Generate integer sequence
epochs = range(1, 21)
```

---
## Step 5 — Plot and label the training and validation loss values

```python
# 绘制折线图 / Draw line plot
plt.plot(epochs, train_values, label='Training Loss')
# 绘制折线图 / Draw line plot
plt.plot(epochs, val_values, label='Validation Loss')
```

---
## Step 6 — Add in a title and axes labels

```python
# 设置图表标题 / Set chart title
plt.title('Training and Validation Loss')
# 设置X轴标签 / Set X-axis label
plt.xlabel('Epochs')
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('Loss')
```

---
## Step 7 — Set the tick locations

```python
# 生成整数序列 / Generate integer sequence
plt.xticks(arange(0, 21, 2))
```

---
## Step 8 — Display the plot

```python
# 显示图例 / Show legend
plt.legend(loc='best')
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Load the training and validation loss dictionaries 是机器学习中的常用技术。  
  *Load the training and validation loss dictionaries is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plotting / 03 Plotting
# Complete Code / 完整代码
# ===============================

from pickle import load
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib.pylab import plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange

# Load the training and validation loss dictionaries
train_loss = load(open('train_loss.pkl', 'rb'))
val_loss = load(open('val_loss.pkl', 'rb'))

# Retrieve each dictionary's values
# 转换为NumPy数组 / Convert to NumPy array
train_values = train_loss.values()
# 转换为NumPy数组 / Convert to NumPy array
val_values = val_loss.values()

# Generate a sequence of integers to represent the epoch numbers
# 生成整数序列 / Generate integer sequence
epochs = range(1, 21)

# Plot and label the training and validation loss values
# 绘制折线图 / Draw line plot
plt.plot(epochs, train_values, label='Training Loss')
# 绘制折线图 / Draw line plot
plt.plot(epochs, val_values, label='Validation Loss')

# Add in a title and axes labels
# 设置图表标题 / Set chart title
plt.title('Training and Validation Loss')
# 设置X轴标签 / Set X-axis label
plt.xlabel('Epochs')
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('Loss')

# Set the tick locations
# 生成整数序列 / Generate integer sequence
plt.xticks(arange(0, 21, 2))

# Display the plot
# 显示图例 / Show legend
plt.legend(loc='best')
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 4 of 9

---

### Chapter Summary / 章节总结

# Chapter 21 Summary / 第21章总结

## Theme / 主题: Chapter 21 / Chapter 21

This chapter contains **9 code files** demonstrating chapter 21.

本章包含 **9 个代码文件**，演示Chapter 21。

---
## Evolution / 演化路线

  1. `01_prepare.ipynb` — Prepare
  2. `02_model.ipynb` — Model
  3. `03_plotting.ipynb` — Plotting
  4. `decoder.ipynb` — Decoder
  5. `encoder.ipynb` — Encoder
  6. `model.ipynb` — Model
  7. `multihead_attention.ipynb` — Multihead Attention
  8. `positional_encoding.ipynb` — Positional Encoding
  9. `prepare_dataset.ipynb` — Prepare Dataset

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 21) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 21）是机器学习流水线中的基础构建块。

---

### Decoder



---

### Encoder



---

### Model

# 01 — Model / Model

**Chapter 21 — File 6 of 9 / 第21章 — 第6个文件（共9个）**

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
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import math, cast, float32, linalg, ones, maximum, newaxis
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras import Model
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense

class TransformerModel(Model):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
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
# 全连接层（Keras） / Fully connected layer (Keras)
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
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
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
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import math, cast, float32, linalg, ones, maximum, newaxis
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras import Model
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense

class TransformerModel(Model):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
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
        # 全连接层（Keras） / Fully connected layer (Keras)
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
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
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

➡️ **Next / 下一步**: File 7 of 9

---

### Multihead Attention



---

### Positional Encoding

# 01 — Positional Encoding / Positional Encoding

**Chapter 21 — File 8 of 9 / 第21章 — 第8个文件（共9个）**

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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Embedding, Layer

class PositionEmbeddingFixedWeights(Layer):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
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
        # 创建全零数组 / Create array of zeros
        P = np.zeros((seq_len, d))
        # 生成整数序列 / Generate integer sequence
        for k in range(seq_len):
            # 生成等差数组 / Generate array with step
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P


    def call(self, inputs):
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
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

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Embedding, Layer

class PositionEmbeddingFixedWeights(Layer):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
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
        # 创建全零数组 / Create array of zeros
        P = np.zeros((seq_len, d))
        # 生成整数序列 / Generate integer sequence
        for k in range(seq_len):
            # 生成等差数组 / Generate array with step
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P


    def call(self, inputs):
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_words + embedded_indices
```

---

➡️ **Next / 下一步**: File 9 of 9

---

### Prepare Dataset



---
