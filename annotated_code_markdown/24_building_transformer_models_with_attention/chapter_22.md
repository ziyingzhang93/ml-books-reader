# 注意力与Transformer / Transformer Models with Attention
## Chapter 22

---

### Inference

# 04 — Inference / 04 Inference

**Chapter 22 — File 1 of 8 / 第22章 — 第1个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Define the model parameters**.

本脚本演示 **Define the model parameters**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  📊 评估模型 / Evaluate Model
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — Step 1

```python
from pickle import load
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import Module
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing.sequence import pad_sequences
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import convert_to_tensor, int64, TensorArray, argmax, newaxis, transpose
from model import TransformerModel
```

---
## Step 2 — Define the model parameters

```python
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of model layers' outputs
d_ff = 2048  # Dimensionality of the inner fully connected layer
n = 6  # Number of layers in the encoder stack
```

---
## Step 3 — Define the dataset parameters

```python
enc_seq_length = 7  # Encoder sequence length
dec_seq_length = 12  # Decoder sequence length
enc_vocab_size = 2404  # Encoder vocabulary size
dec_vocab_size = 3864  # Decoder vocabulary size
```

---
## Step 4 — Create model

```python
inferencing_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length,
                                     dec_seq_length, h, d_k, d_v, d_model, d_ff, n, 0)

class Translate(Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, inferencing_model, **kwargs):
        super().__init__(**kwargs)
        self.transformer = inferencing_model

    def load_tokenizer(self, name):
        # 打开文件（自动关闭） / Open file (auto-close)
        with open(name, 'rb') as handle:
            return load(handle)

    def __call__(self, sentence):
```

---
## Step 5 — Append start and end of string tokens to the input sentence

```python
sentence[0] = "<START> " + sentence[0] + " <EOS>"
```

---
## Step 6 — Load encoder and decoder tokenizers

```python
enc_tokenizer = self.load_tokenizer('enc_tokenizer.pkl')
        dec_tokenizer = self.load_tokenizer('dec_tokenizer.pkl')
```

---
## Step 7 — Prepare the input sentence by tokenizing, padding and converting to tensor

```python
encoder_input = enc_tokenizer.texts_to_sequences(sentence)
        encoder_input = pad_sequences(encoder_input,
                                      maxlen=enc_seq_length, padding='post')
        encoder_input = convert_to_tensor(encoder_input, dtype=int64)
```

---
## Step 8 — Prepare the output <START> token by tokenizing, and converting to tensor

```python
output_start = dec_tokenizer.texts_to_sequences(["<START>"])
        output_start = convert_to_tensor(output_start[0], dtype=int64)
```

---
## Step 9 — Prepare the output <EOS> token by tokenizing, and converting to tensor

```python
output_end = dec_tokenizer.texts_to_sequences(["<EOS>"])
        output_end = convert_to_tensor(output_end[0], dtype=int64)
```

---
## Step 10 — Prepare the output array of dynamic size

```python
decoder_output = TensorArray(dtype=int64, size=0, dynamic_size=True)
        decoder_output = decoder_output.write(0, output_start)

        # 生成整数序列 / Generate integer sequence
        for i in range(dec_seq_length):
```

---
## Step 11 — Predict an output token

```python
prediction = self.transformer(encoder_input,transpose(decoder_output.stack()),
                                          training=False)
            prediction = prediction[:, -1, :]
```

---
## Step 12 — Select the prediction with the highest score

```python
predicted_id = argmax(prediction, axis=-1)
            predicted_id = predicted_id[0][newaxis]
```

---
## Step 13 — Write the selected prediction to the output array at the next
available index

```python
decoder_output = decoder_output.write(i + 1, predicted_id)
```

---
## Step 14 — Break if an <EOS> token is predicted

```python
if predicted_id == output_end:
                break

        output = transpose(decoder_output.stack())[0]
        output = output.numpy()

        output_str = []
```

---
## Step 15 — Decode the predicted tokens into an output string

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
for i in range(output.shape[0]):
            key = output[i]
            # 添加元素到列表末尾 / Append element to list end
            output_str.append(dec_tokenizer.index_word[key])

        return output_str
```

---
## Learning Notes / 学习笔记

- **概念**: Define the model parameters 是机器学习中的常用技术。  
  *Define the model parameters is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Inference / 04 Inference
# Complete Code / 完整代码
# ===============================

from pickle import load
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import Module
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing.sequence import pad_sequences
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import convert_to_tensor, int64, TensorArray, argmax, newaxis, transpose
from model import TransformerModel

# Define the model parameters
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of model layers' outputs
d_ff = 2048  # Dimensionality of the inner fully connected layer
n = 6  # Number of layers in the encoder stack

# Define the dataset parameters
enc_seq_length = 7  # Encoder sequence length
dec_seq_length = 12  # Decoder sequence length
enc_vocab_size = 2404  # Encoder vocabulary size
dec_vocab_size = 3864  # Decoder vocabulary size

# Create model
inferencing_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length,
                                     dec_seq_length, h, d_k, d_v, d_model, d_ff, n, 0)

class Translate(Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, inferencing_model, **kwargs):
        super().__init__(**kwargs)
        self.transformer = inferencing_model

    def load_tokenizer(self, name):
        # 打开文件（自动关闭） / Open file (auto-close)
        with open(name, 'rb') as handle:
            return load(handle)

    def __call__(self, sentence):
        # Append start and end of string tokens to the input sentence
        sentence[0] = "<START> " + sentence[0] + " <EOS>"

        # Load encoder and decoder tokenizers
        enc_tokenizer = self.load_tokenizer('enc_tokenizer.pkl')
        dec_tokenizer = self.load_tokenizer('dec_tokenizer.pkl')

        # Prepare the input sentence by tokenizing, padding and converting to tensor
        encoder_input = enc_tokenizer.texts_to_sequences(sentence)
        encoder_input = pad_sequences(encoder_input,
                                      maxlen=enc_seq_length, padding='post')
        encoder_input = convert_to_tensor(encoder_input, dtype=int64)

        # Prepare the output <START> token by tokenizing, and converting to tensor
        output_start = dec_tokenizer.texts_to_sequences(["<START>"])
        output_start = convert_to_tensor(output_start[0], dtype=int64)

        # Prepare the output <EOS> token by tokenizing, and converting to tensor
        output_end = dec_tokenizer.texts_to_sequences(["<EOS>"])
        output_end = convert_to_tensor(output_end[0], dtype=int64)

        # Prepare the output array of dynamic size
        decoder_output = TensorArray(dtype=int64, size=0, dynamic_size=True)
        decoder_output = decoder_output.write(0, output_start)

        # 生成整数序列 / Generate integer sequence
        for i in range(dec_seq_length):
            # Predict an output token
            prediction = self.transformer(encoder_input,transpose(decoder_output.stack()),
                                          training=False)
            prediction = prediction[:, -1, :]

            # Select the prediction with the highest score
            predicted_id = argmax(prediction, axis=-1)
            predicted_id = predicted_id[0][newaxis]

            # Write the selected prediction to the output array at the next
            # available index
            decoder_output = decoder_output.write(i + 1, predicted_id)

            # Break if an <EOS> token is predicted
            if predicted_id == output_end:
                break

        output = transpose(decoder_output.stack())[0]
        output = output.numpy()

        output_str = []

        # Decode the predicted tokens into an output string
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        for i in range(output.shape[0]):
            key = output[i]
            # 添加元素到列表末尾 / Append element to list end
            output_str.append(dec_tokenizer.index_word[key])

        return output_str
```

---

➡️ **Next / 下一步**: File 2 of 8

---

### Translate

# 06 — Translate / 06 Translate

**Chapter 22 — File 2 of 8 / 第22章 — 第2个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Define the model parameters**.

本脚本演示 **Define the model parameters**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  📊 评估模型 / Evaluate Model
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — Step 1

```python
from pickle import load
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import Module
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing.sequence import pad_sequences
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import convert_to_tensor, int64, TensorArray, argmax, newaxis, transpose
from model import TransformerModel
```

---
## Step 2 — Define the model parameters

```python
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of model layers' outputs
d_ff = 2048  # Dimensionality of the inner fully connected layer
n = 6  # Number of layers in the encoder stack
```

---
## Step 3 — Define the dataset parameters

```python
enc_seq_length = 7  # Encoder sequence length
dec_seq_length = 12  # Decoder sequence length
enc_vocab_size = 2404  # Encoder vocabulary size
dec_vocab_size = 3864  # Decoder vocabulary size
```

---
## Step 4 — Create model

```python
inferencing_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length,
                                     dec_seq_length, h, d_k, d_v, d_model, d_ff, n, 0)

class Translate(Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, inferencing_model, **kwargs):
        super().__init__(**kwargs)
        self.transformer = inferencing_model

    def load_tokenizer(self, name):
        # 打开文件（自动关闭） / Open file (auto-close)
        with open(name, 'rb') as handle:
            return load(handle)

    def __call__(self, sentence):
```

---
## Step 5 — Append start and end of string tokens to the input sentence

```python
sentence[0] = "<START> " + sentence[0] + " <EOS>"
```

---
## Step 6 — Load encoder and decoder tokenizers

```python
enc_tokenizer = self.load_tokenizer('enc_tokenizer.pkl')
        dec_tokenizer = self.load_tokenizer('dec_tokenizer.pkl')
```

---
## Step 7 — Prepare the input sentence by tokenizing, padding and converting to tensor

```python
encoder_input = enc_tokenizer.texts_to_sequences(sentence)
        encoder_input = pad_sequences(encoder_input,
                                      maxlen=enc_seq_length, padding='post')
        encoder_input = convert_to_tensor(encoder_input, dtype=int64)
```

---
## Step 8 — Prepare the output <START> token by tokenizing, and converting to tensor

```python
output_start = dec_tokenizer.texts_to_sequences(["<START>"])
        output_start = convert_to_tensor(output_start[0], dtype=int64)
```

---
## Step 9 — Prepare the output <EOS> token by tokenizing, and converting to tensor

```python
output_end = dec_tokenizer.texts_to_sequences(["<EOS>"])
        output_end = convert_to_tensor(output_end[0], dtype=int64)
```

---
## Step 10 — Prepare the output array of dynamic size

```python
decoder_output = TensorArray(dtype=int64, size=0, dynamic_size=True)
        decoder_output = decoder_output.write(0, output_start)

        # 生成整数序列 / Generate integer sequence
        for i in range(dec_seq_length):
```

---
## Step 11 — Predict an output token

```python
prediction = self.transformer(encoder_input,transpose(decoder_output.stack()),
                                          training=False)
            prediction = prediction[:, -1, :]
```

---
## Step 12 — Select the prediction with the highest score

```python
predicted_id = argmax(prediction, axis=-1)
            predicted_id = predicted_id[0][newaxis]
```

---
## Step 13 — Write the selected prediction to the output array at the next
available index

```python
decoder_output = decoder_output.write(i + 1, predicted_id)
```

---
## Step 14 — Break if an <EOS> token is predicted

```python
if predicted_id == output_end:
                break

        output = transpose(decoder_output.stack())[0]
        output = output.numpy()

        output_str = []
```

---
## Step 15 — Decode the predicted tokens into an output string

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
for i in range(output.shape[0]):
            key = output[i]
            # 添加元素到列表末尾 / Append element to list end
            output_str.append(dec_tokenizer.index_word[key])

        return output_str
```

---
## Step 16 — Sentence to translate

```python
sentence = ['im thirsty']
```

---
## Step 17 — Load the trained model's weights at the specified epoch

```python
inferencing_model.load_weights('weights/wghts16.ckpt')
```

---
## Step 18 — Create a new instance of the 'Translate' class

```python
translator = Translate(inferencing_model)
```

---
## Step 19 — Translate the input sentence

```python
# 打印输出 / Print output
print(translator(sentence))
```

---
## Learning Notes / 学习笔记

- **概念**: Define the model parameters 是机器学习中的常用技术。  
  *Define the model parameters is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Translate / 06 Translate
# Complete Code / 完整代码
# ===============================

from pickle import load
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import Module
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing.sequence import pad_sequences
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import convert_to_tensor, int64, TensorArray, argmax, newaxis, transpose
from model import TransformerModel

# Define the model parameters
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of model layers' outputs
d_ff = 2048  # Dimensionality of the inner fully connected layer
n = 6  # Number of layers in the encoder stack

# Define the dataset parameters
enc_seq_length = 7  # Encoder sequence length
dec_seq_length = 12  # Decoder sequence length
enc_vocab_size = 2404  # Encoder vocabulary size
dec_vocab_size = 3864  # Decoder vocabulary size

# Create model
inferencing_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length,
                                     dec_seq_length, h, d_k, d_v, d_model, d_ff, n, 0)

class Translate(Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, inferencing_model, **kwargs):
        super().__init__(**kwargs)
        self.transformer = inferencing_model

    def load_tokenizer(self, name):
        # 打开文件（自动关闭） / Open file (auto-close)
        with open(name, 'rb') as handle:
            return load(handle)

    def __call__(self, sentence):
        # Append start and end of string tokens to the input sentence
        sentence[0] = "<START> " + sentence[0] + " <EOS>"

        # Load encoder and decoder tokenizers
        enc_tokenizer = self.load_tokenizer('enc_tokenizer.pkl')
        dec_tokenizer = self.load_tokenizer('dec_tokenizer.pkl')

        # Prepare the input sentence by tokenizing, padding and converting to tensor
        encoder_input = enc_tokenizer.texts_to_sequences(sentence)
        encoder_input = pad_sequences(encoder_input,
                                      maxlen=enc_seq_length, padding='post')
        encoder_input = convert_to_tensor(encoder_input, dtype=int64)

        # Prepare the output <START> token by tokenizing, and converting to tensor
        output_start = dec_tokenizer.texts_to_sequences(["<START>"])
        output_start = convert_to_tensor(output_start[0], dtype=int64)

        # Prepare the output <EOS> token by tokenizing, and converting to tensor
        output_end = dec_tokenizer.texts_to_sequences(["<EOS>"])
        output_end = convert_to_tensor(output_end[0], dtype=int64)

        # Prepare the output array of dynamic size
        decoder_output = TensorArray(dtype=int64, size=0, dynamic_size=True)
        decoder_output = decoder_output.write(0, output_start)

        # 生成整数序列 / Generate integer sequence
        for i in range(dec_seq_length):
            # Predict an output token
            prediction = self.transformer(encoder_input,transpose(decoder_output.stack()),
                                          training=False)
            prediction = prediction[:, -1, :]

            # Select the prediction with the highest score
            predicted_id = argmax(prediction, axis=-1)
            predicted_id = predicted_id[0][newaxis]

            # Write the selected prediction to the output array at the next
            # available index
            decoder_output = decoder_output.write(i + 1, predicted_id)

            # Break if an <EOS> token is predicted
            if predicted_id == output_end:
                break

        output = transpose(decoder_output.stack())[0]
        output = output.numpy()

        output_str = []

        # Decode the predicted tokens into an output string
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        for i in range(output.shape[0]):
            key = output[i]
            # 添加元素到列表末尾 / Append element to list end
            output_str.append(dec_tokenizer.index_word[key])

        return output_str

# Sentence to translate
sentence = ['im thirsty']

# Load the trained model's weights at the specified epoch
inferencing_model.load_weights('weights/wghts16.ckpt')

# Create a new instance of the 'Translate' class
translator = Translate(inferencing_model)

# Translate the input sentence
# 打印输出 / Print output
print(translator(sentence))
```

---

➡️ **Next / 下一步**: File 3 of 8

---

### Chapter Summary / 章节总结

# Chapter 22 Summary / 第22章总结

## Theme / 主题: Chapter 22 / Chapter 22

This chapter contains **8 code files** demonstrating chapter 22.

本章包含 **8 个代码文件**，演示Chapter 22。

---
## Evolution / 演化路线

  1. `04_inference.ipynb` — Inference
  2. `06_translate.ipynb` — Translate
  3. `decoder.ipynb` — Decoder
  4. `encoder.ipynb` — Encoder
  5. `model.ipynb` — Model
  6. `multihead_attention.ipynb` — Multihead Attention
  7. `positional_encoding.ipynb` — Positional Encoding
  8. `prepare_dataset.ipynb` — Prepare Dataset

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 22) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 22）是机器学习流水线中的基础构建块。

---

### Decoder

# 01 — Decoder / Decoder

**Chapter 22 — File 3 of 8 / 第22章 — 第3个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Implementing the Decoder Layer**.

本脚本演示 **Implementing the Decoder Layer**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Layer, Dropout
from multihead_attention import MultiHeadAttention
from positional_encoding import PositionEmbeddingFixedWeights
from encoder import AddNormalization, FeedForward
```

---
## Step 2 — Implementing the Decoder Layer

```python
class DecoderLayer(Layer):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super().__init__(**kwargs)
        self.multihead_attention1 = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.multihead_attention2 = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout3 = Dropout(rate)
        self.add_norm3 = AddNormalization()

    def call(self, x, encoder_output, lookahead_mask, padding_mask, training):
```

---
## Step 3 — Multi-head attention layer

```python
multihead_output1 = self.multihead_attention1(x, x, x, lookahead_mask)
```

---
## Step 4 — Expected output shape = (batch_size, sequence_length, d_model)
Add in a dropout layer

```python
multihead_output1 = self.dropout1(multihead_output1, training=training)
```

---
## Step 5 — Followed by an Add & Norm layer

```python
addnorm_output1 = self.add_norm1(x, multihead_output1)
```

---
## Step 6 — Expected output shape = (batch_size, sequence_length, d_model)
Followed by another multi-head attention layer

```python
multihead_output2 = self.multihead_attention2(addnorm_output1, encoder_output,
                                                      encoder_output, padding_mask)
```

---
## Step 7 — Add in another dropout layer

```python
multihead_output2 = self.dropout2(multihead_output2, training=training)
```

---
## Step 8 — Followed by another Add & Norm layer

```python
addnorm_output2 = self.add_norm1(addnorm_output1, multihead_output2)
```

---
## Step 9 — Followed by a fully connected layer

```python
feedforward_output = self.feed_forward(addnorm_output2)
```

---
## Step 10 — Expected output shape = (batch_size, sequence_length, d_model)
Add in another dropout layer

```python
feedforward_output = self.dropout3(feedforward_output, training=training)
```

---
## Step 11 — Followed by another Add & Norm layer

```python
return self.add_norm3(addnorm_output2, feedforward_output)
```

---
## Step 12 — Implementing the Decoder

```python
class Decoder(Layer):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate,
                       **kwargs):
        super().__init__(**kwargs)
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size,
                                                          d_model)
        self.dropout = Dropout(rate)
        self.decoder_layer = [DecoderLayer(h, d_k, d_v, d_model, d_ff, rate)
                              # 生成整数序列 / Generate integer sequence
                              for _ in range(n)]

    def call(self, output_target, encoder_output, lookahead_mask, padding_mask, training):
```

---
## Step 13 — Generate the positional encoding

```python
pos_encoding_output = self.pos_encoding(output_target)
```

---
## Step 14 — Expected output shape = (number of sentences, sequence_length, d_model)
Add in a dropout layer

```python
x = self.dropout(pos_encoding_output, training=training)
```

---
## Step 15 — Pass on the positional encoded values to each encoder layer

```python
# 同时获取索引和值 / Get both index and value
for i, layer in enumerate(self.decoder_layer):
            x = layer(x, encoder_output, lookahead_mask, padding_mask, training)

        return x
```

---
## Learning Notes / 学习笔记

- **概念**: Implementing the Decoder Layer 是机器学习中的常用技术。  
  *Implementing the Decoder Layer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Decoder / Decoder
# Complete Code / 完整代码
# ===============================

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Layer, Dropout
from multihead_attention import MultiHeadAttention
from positional_encoding import PositionEmbeddingFixedWeights
from encoder import AddNormalization, FeedForward

# Implementing the Decoder Layer
class DecoderLayer(Layer):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super().__init__(**kwargs)
        self.multihead_attention1 = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.multihead_attention2 = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout3 = Dropout(rate)
        self.add_norm3 = AddNormalization()

    def call(self, x, encoder_output, lookahead_mask, padding_mask, training):
        # Multi-head attention layer
        multihead_output1 = self.multihead_attention1(x, x, x, lookahead_mask)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        multihead_output1 = self.dropout1(multihead_output1, training=training)

        # Followed by an Add & Norm layer
        addnorm_output1 = self.add_norm1(x, multihead_output1)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Followed by another multi-head attention layer
        multihead_output2 = self.multihead_attention2(addnorm_output1, encoder_output,
                                                      encoder_output, padding_mask)

        # Add in another dropout layer
        multihead_output2 = self.dropout2(multihead_output2, training=training)

        # Followed by another Add & Norm layer
        addnorm_output2 = self.add_norm1(addnorm_output1, multihead_output2)

        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output2)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in another dropout layer
        feedforward_output = self.dropout3(feedforward_output, training=training)

        # Followed by another Add & Norm layer
        return self.add_norm3(addnorm_output2, feedforward_output)

# Implementing the Decoder
class Decoder(Layer):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate,
                       **kwargs):
        super().__init__(**kwargs)
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size,
                                                          d_model)
        self.dropout = Dropout(rate)
        self.decoder_layer = [DecoderLayer(h, d_k, d_v, d_model, d_ff, rate)
                              # 生成整数序列 / Generate integer sequence
                              for _ in range(n)]

    def call(self, output_target, encoder_output, lookahead_mask, padding_mask, training):
        # Generate the positional encoding
        pos_encoding_output = self.pos_encoding(output_target)
        # Expected output shape = (number of sentences, sequence_length, d_model)

        # Add in a dropout layer
        x = self.dropout(pos_encoding_output, training=training)

        # Pass on the positional encoded values to each encoder layer
        # 同时获取索引和值 / Get both index and value
        for i, layer in enumerate(self.decoder_layer):
            x = layer(x, encoder_output, lookahead_mask, padding_mask, training)

        return x
```

---

➡️ **Next / 下一步**: File 4 of 8

---

### Encoder

# 01 — Encoder / 数据编码

**Chapter 22 — File 4 of 8 / 第22章 — 第4个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Implementing the Add & Norm Layer**.

本脚本演示 **Implementing the Add & Norm Layer**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LayerNormalization, Layer, Dense, ReLU, Dropout
from multihead_attention import MultiHeadAttention
from positional_encoding import PositionEmbeddingFixedWeights
```

---
## Step 2 — Implementing the Add & Norm Layer

```python
class AddNormalization(Layer):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
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
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, d_ff, d_model, **kwargs):
        super().__init__(**kwargs)
        # 全连接层（Keras） / Fully connected layer (Keras)
        self.fully_connected1 = Dense(d_ff)  # First fully connected layer
        # 全连接层（Keras） / Fully connected layer (Keras)
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
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
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
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate,
                       **kwargs):
        super().__init__(**kwargs)
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size,
                                                          d_model)
        self.dropout = Dropout(rate)
        self.encoder_layer = [EncoderLayer(h, d_k, d_v, d_model, d_ff, rate)
                              # 生成整数序列 / Generate integer sequence
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
# 同时获取索引和值 / Get both index and value
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

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Encoder / 数据编码
# Complete Code / 完整代码
# ===============================

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LayerNormalization, Layer, Dense, ReLU, Dropout
from multihead_attention import MultiHeadAttention
from positional_encoding import PositionEmbeddingFixedWeights

# Implementing the Add & Norm Layer
class AddNormalization(Layer):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
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
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, d_ff, d_model, **kwargs):
        super().__init__(**kwargs)
        # 全连接层（Keras） / Fully connected layer (Keras)
        self.fully_connected1 = Dense(d_ff)  # First fully connected layer
        # 全连接层（Keras） / Fully connected layer (Keras)
        self.fully_connected2 = Dense(d_model)  # Second fully connected layer
        self.activation = ReLU()  # ReLU activation layer

    def call(self, x):
        # The input is passed into the two fully-connected layers, with a ReLU in between
        x_fc1 = self.fully_connected1(x)

        return self.fully_connected2(self.activation(x_fc1))

# Implementing the Encoder Layer
class EncoderLayer(Layer):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
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
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate,
                       **kwargs):
        super().__init__(**kwargs)
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size,
                                                          d_model)
        self.dropout = Dropout(rate)
        self.encoder_layer = [EncoderLayer(h, d_k, d_v, d_model, d_ff, rate)
                              # 生成整数序列 / Generate integer sequence
                              for _ in range(n)]

    def call(self, input_sentence, padding_mask, training):
        # Generate the positional encoding
        pos_encoding_output = self.pos_encoding(input_sentence)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        x = self.dropout(pos_encoding_output, training=training)

        # Pass on the positional encoded values to each encoder layer
        # 同时获取索引和值 / Get both index and value
        for i, layer in enumerate(self.encoder_layer):
            x = layer(x, padding_mask, training)

        return x
```

---

➡️ **Next / 下一步**: File 5 of 8

---

### Model



---

### Multihead Attention

# 01 — Multihead Attention / 注意力机制

**Chapter 22 — File 6 of 8 / 第22章 — 第6个文件（共8个）**

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
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Step 1

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import math, matmul, reshape, shape, transpose, cast, float32
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, Layer
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.backend import softmax
```

---
## Step 2 — Implementing the Scaled-Dot Product Attention

```python
class DotProductAttention(Layer):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
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
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, h, d_k, d_v, d_model, **kwargs):
        super().__init__(**kwargs)
        self.attention = DotProductAttention()  # Scaled dot product attention
        self.heads = h  # Number of attention heads to use
        self.d_k = d_k  # Dimensionality of the linearly projected queries and keys
        self.d_v = d_v  # Dimensionality of the linearly projected values
        self.d_model = d_model  # Dimensionality of the model
        # 全连接层（Keras） / Fully connected layer (Keras)
        self.W_q = Dense(d_k)   # Learned projection matrix for the queries
        # 全连接层（Keras） / Fully connected layer (Keras)
        self.W_k = Dense(d_k)   # Learned projection matrix for the keys
        # 全连接层（Keras） / Fully connected layer (Keras)
        self.W_v = Dense(d_v)   # Learned projection matrix for the values
        # 全连接层（Keras） / Fully connected layer (Keras)
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

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import math, matmul, reshape, shape, transpose, cast, float32
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, Layer
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.backend import softmax

# Implementing the Scaled-Dot Product Attention
class DotProductAttention(Layer):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
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
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, h, d_k, d_v, d_model, **kwargs):
        super().__init__(**kwargs)
        self.attention = DotProductAttention()  # Scaled dot product attention
        self.heads = h  # Number of attention heads to use
        self.d_k = d_k  # Dimensionality of the linearly projected queries and keys
        self.d_v = d_v  # Dimensionality of the linearly projected values
        self.d_model = d_model  # Dimensionality of the model
        # 全连接层（Keras） / Fully connected layer (Keras)
        self.W_q = Dense(d_k)   # Learned projection matrix for the queries
        # 全连接层（Keras） / Fully connected layer (Keras)
        self.W_k = Dense(d_k)   # Learned projection matrix for the keys
        # 全连接层（Keras） / Fully connected layer (Keras)
        self.W_v = Dense(d_v)   # Learned projection matrix for the values
        # 全连接层（Keras） / Fully connected layer (Keras)
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

➡️ **Next / 下一步**: File 7 of 8

---

### Positional Encoding

# 01 — Positional Encoding / Positional Encoding

**Chapter 22 — File 7 of 8 / 第22章 — 第7个文件（共8个）**

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

➡️ **Next / 下一步**: File 8 of 8

---

### Prepare Dataset

# 01 — Prepare Dataset / 数据准备

**Chapter 22 — File 8 of 8 / 第22章 — 第8个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Fit a tokenizer**.

本脚本演示 **Fit a tokenizer**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — Step 1

```python
from pickle import load, dump, HIGHEST_PROTOCOL
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import shuffle
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import savetxt
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing.text import Tokenizer
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing.sequence import pad_sequences
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import convert_to_tensor, int64

class PrepareDataset:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_sentences = 15000  # Number of sentences to include in the dataset
        self.train_split = 0.8  # Ratio of the training data split
        self.val_split = 0.1  # Ratio of the validation data split
```

---
## Step 2 — Fit a tokenizer

```python
def create_tokenizer(self, dataset):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(dataset)

        return tokenizer

    def find_seq_length(self, dataset):
        # 获取长度 / Get length
        return max(len(seq.split()) for seq in dataset)

    def find_vocab_size(self, tokenizer, dataset):
        tokenizer.fit_on_texts(dataset)

        # 获取长度 / Get length
        return len(tokenizer.word_index) + 1
```

---
## Step 3 — Encode and pad the input sequences

```python
def encode_pad(self, dataset, tokenizer, seq_length):
        x = tokenizer.texts_to_sequences(dataset)
        x = pad_sequences(x, maxlen=seq_length, padding='post')
        x = convert_to_tensor(x, dtype=int64)

        return x

    def save_tokenizer(self, tokenizer, name):
        # 打开文件（自动关闭） / Open file (auto-close)
        with open(name + '_tokenizer.pkl', 'wb') as handle:
            dump(tokenizer, handle, protocol=HIGHEST_PROTOCOL)

    def __call__(self, filename, **kwargs):
```

---
## Step 4 — Load a clean dataset

```python
clean_dataset = load(open(filename, 'rb'))
```

---
## Step 5 — Reduce dataset size

```python
dataset = clean_dataset[:self.n_sentences, :]
```

---
## Step 6 — Include start and end of string tokens

```python
# 生成整数序列 / Generate integer sequence
for i in range(dataset[:, 0].size):
            dataset[i, 0] = "<START> " + dataset[i, 0] + " <EOS>"
            dataset[i, 1] = "<START> " + dataset[i, 1] + " <EOS>"
```

---
## Step 7 — Random shuffle the dataset

```python
shuffle(dataset)
```

---
## Step 8 — Split the dataset in training, validation and test sets

```python
train = dataset[:int(self.n_sentences * self.train_split)]
        val = dataset[int(self.n_sentences * self.train_split):
                      int(self.n_sentences * (1-self.val_split))]
        test = dataset[int(self.n_sentences * (1 - self.val_split)):]
```

---
## Step 9 — Prepare tokenizer for the encoder input

```python
enc_tokenizer = self.create_tokenizer(dataset[:, 0])
        enc_seq_length = self.find_seq_length(dataset[:, 0])
        enc_vocab_size = self.find_vocab_size(enc_tokenizer, train[:, 0])
```

---
## Step 10 — Prepare tokenizer for the decoder input

```python
dec_tokenizer = self.create_tokenizer(dataset[:, 1])
        dec_seq_length = self.find_seq_length(dataset[:, 1])
        dec_vocab_size = self.find_vocab_size(dec_tokenizer, train[:, 1])
```

---
## Step 11 — Encode and pad the training input

```python
trainX = self.encode_pad(train[:, 0], enc_tokenizer, enc_seq_length)
        trainY = self.encode_pad(train[:, 1], dec_tokenizer, dec_seq_length)
```

---
## Step 12 — Encode and pad the validation input

```python
valX = self.encode_pad(val[:, 0], enc_tokenizer, enc_seq_length)
        valY = self.encode_pad(val[:, 1], dec_tokenizer, dec_seq_length)
```

---
## Step 13 — Save the encoder tokenizer

```python
self.save_tokenizer(enc_tokenizer, 'enc')
```

---
## Step 14 — Save the decoder tokenizer

```python
self.save_tokenizer(dec_tokenizer, 'dec')
```

---
## Step 15 — Save the testing dataset into a text file

```python
savetxt('test_dataset.txt', test, fmt='%s')

        return (trainX, trainY, valX, valY, train, val, enc_seq_length,
                dec_seq_length, enc_vocab_size, dec_vocab_size)
```

---
## Learning Notes / 学习笔记

- **概念**: Fit a tokenizer 是机器学习中的常用技术。  
  *Fit a tokenizer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Prepare Dataset / 数据准备
# Complete Code / 完整代码
# ===============================

from pickle import load, dump, HIGHEST_PROTOCOL
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import shuffle
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import savetxt
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing.text import Tokenizer
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing.sequence import pad_sequences
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import convert_to_tensor, int64

class PrepareDataset:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_sentences = 15000  # Number of sentences to include in the dataset
        self.train_split = 0.8  # Ratio of the training data split
        self.val_split = 0.1  # Ratio of the validation data split

    # Fit a tokenizer
    def create_tokenizer(self, dataset):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(dataset)

        return tokenizer

    def find_seq_length(self, dataset):
        # 获取长度 / Get length
        return max(len(seq.split()) for seq in dataset)

    def find_vocab_size(self, tokenizer, dataset):
        tokenizer.fit_on_texts(dataset)

        # 获取长度 / Get length
        return len(tokenizer.word_index) + 1

    # Encode and pad the input sequences
    def encode_pad(self, dataset, tokenizer, seq_length):
        x = tokenizer.texts_to_sequences(dataset)
        x = pad_sequences(x, maxlen=seq_length, padding='post')
        x = convert_to_tensor(x, dtype=int64)

        return x

    def save_tokenizer(self, tokenizer, name):
        # 打开文件（自动关闭） / Open file (auto-close)
        with open(name + '_tokenizer.pkl', 'wb') as handle:
            dump(tokenizer, handle, protocol=HIGHEST_PROTOCOL)

    def __call__(self, filename, **kwargs):
        # Load a clean dataset
        clean_dataset = load(open(filename, 'rb'))

        # Reduce dataset size
        dataset = clean_dataset[:self.n_sentences, :]

        # Include start and end of string tokens
        # 生成整数序列 / Generate integer sequence
        for i in range(dataset[:, 0].size):
            dataset[i, 0] = "<START> " + dataset[i, 0] + " <EOS>"
            dataset[i, 1] = "<START> " + dataset[i, 1] + " <EOS>"

        # Random shuffle the dataset
        shuffle(dataset)

        # Split the dataset in training, validation and test sets
        train = dataset[:int(self.n_sentences * self.train_split)]
        val = dataset[int(self.n_sentences * self.train_split):
                      int(self.n_sentences * (1-self.val_split))]
        test = dataset[int(self.n_sentences * (1 - self.val_split)):]

        # Prepare tokenizer for the encoder input
        enc_tokenizer = self.create_tokenizer(dataset[:, 0])
        enc_seq_length = self.find_seq_length(dataset[:, 0])
        enc_vocab_size = self.find_vocab_size(enc_tokenizer, train[:, 0])

        # Prepare tokenizer for the decoder input
        dec_tokenizer = self.create_tokenizer(dataset[:, 1])
        dec_seq_length = self.find_seq_length(dataset[:, 1])
        dec_vocab_size = self.find_vocab_size(dec_tokenizer, train[:, 1])

        # Encode and pad the training input
        trainX = self.encode_pad(train[:, 0], enc_tokenizer, enc_seq_length)
        trainY = self.encode_pad(train[:, 1], dec_tokenizer, dec_seq_length)

        # Encode and pad the validation input
        valX = self.encode_pad(val[:, 0], enc_tokenizer, enc_seq_length)
        valY = self.encode_pad(val[:, 1], dec_tokenizer, dec_seq_length)

        # Save the encoder tokenizer
        self.save_tokenizer(enc_tokenizer, 'enc')

        # Save the decoder tokenizer
        self.save_tokenizer(dec_tokenizer, 'dec')

        # Save the testing dataset into a text file
        savetxt('test_dataset.txt', test, fmt='%s')

        return (trainX, trainY, valX, valY, train, val, enc_seq_length,
                dec_seq_length, enc_vocab_size, dec_vocab_size)
```

---
