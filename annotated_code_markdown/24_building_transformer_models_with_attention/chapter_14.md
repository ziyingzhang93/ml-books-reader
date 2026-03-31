# 注意力与Transformer / Transformer Models with Attention
## Chapter 14

---

### Vectorize

# 02 — Vectorize / 02 Vectorize

**Chapter 14 — File 1 of 8 / 第14章 — 第1个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Create the TextVectorization layer**.

本脚本演示 **Create the TextVectorization layer**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import convert_to_tensor
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import TextVectorization
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.data import Dataset

output_sequence_length = 5
vocab_size = 10
sentences = [["I am a robot"], ["you too robot"]]
sentence_data = Dataset.from_tensor_slices(sentences)
```

---
## Step 2 — Create the TextVectorization layer

```python
vectorize_layer = TextVectorization(output_sequence_length=output_sequence_length,
                                    max_tokens=vocab_size)
```

---
## Step 3 — Train the layer to create a dictionary

```python
vectorize_layer.adapt(sentence_data)
```

---
## Step 4 — Convert all sentences to tensors

```python
word_tensors = convert_to_tensor(sentences, dtype=tf.string)
```

---
## Step 5 — Use the word tensors to get vectorized phrases

```python
vectorized_words = vectorize_layer(word_tensors)
# 打印输出 / Print output
print("Vocabulary: ", vectorize_layer.get_vocabulary())
# 打印输出 / Print output
print("Vectorized words: ", vectorized_words)
```

---
## Learning Notes / 学习笔记

- **概念**: Create the TextVectorization layer 是机器学习中的常用技术。  
  *Create the TextVectorization layer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Vectorize / 02 Vectorize
# Complete Code / 完整代码
# ===============================

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import convert_to_tensor
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import TextVectorization
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.data import Dataset

output_sequence_length = 5
vocab_size = 10
sentences = [["I am a robot"], ["you too robot"]]
sentence_data = Dataset.from_tensor_slices(sentences)
# Create the TextVectorization layer
vectorize_layer = TextVectorization(output_sequence_length=output_sequence_length,
                                    max_tokens=vocab_size)
# Train the layer to create a dictionary
vectorize_layer.adapt(sentence_data)
# Convert all sentences to tensors
word_tensors = convert_to_tensor(sentences, dtype=tf.string)
# Use the word tensors to get vectorized phrases
vectorized_words = vectorize_layer(word_tensors)
# 打印输出 / Print output
print("Vocabulary: ", vectorize_layer.get_vocabulary())
# 打印输出 / Print output
print("Vectorized words: ", vectorized_words)
```

---

➡️ **Next / 下一步**: File 2 of 8

---

### Embedding

# 03 — Embedding / 词嵌入

**Chapter 14 — File 2 of 8 / 第14章 — 第2个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Create the TextVectorization layer**.

本脚本演示 **Create the TextVectorization layer**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import convert_to_tensor
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import TextVectorization, Embedding
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.data import Dataset

output_length = 6
output_sequence_length = 5
vocab_size = 10
sentences = [["I am a robot"], ["you too robot"]]

sentence_data = Dataset.from_tensor_slices(sentences)
```

---
## Step 2 — Create the TextVectorization layer

```python
vectorize_layer = TextVectorization(output_sequence_length=output_sequence_length,
                                    max_tokens=vocab_size)
```

---
## Step 3 — Train the layer to create a dictionary

```python
vectorize_layer.adapt(sentence_data)
```

---
## Step 4 — Convert all sentences to tensors

```python
word_tensors = convert_to_tensor(sentences, dtype=tf.string)
```

---
## Step 5 — Use the word tensors to get vectorized phrases

```python
vectorized_words = vectorize_layer(word_tensors)

word_embedding_layer = Embedding(vocab_size, output_length)
embedded_words = word_embedding_layer(vectorized_words)
# 打印输出 / Print output
print(embedded_words)
```

---
## Learning Notes / 学习笔记

- **概念**: Create the TextVectorization layer 是机器学习中的常用技术。  
  *Create the TextVectorization layer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Embedding / 词嵌入
# Complete Code / 完整代码
# ===============================

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import convert_to_tensor
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import TextVectorization, Embedding
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.data import Dataset

output_length = 6
output_sequence_length = 5
vocab_size = 10
sentences = [["I am a robot"], ["you too robot"]]

sentence_data = Dataset.from_tensor_slices(sentences)
# Create the TextVectorization layer
vectorize_layer = TextVectorization(output_sequence_length=output_sequence_length,
                                    max_tokens=vocab_size)
# Train the layer to create a dictionary
vectorize_layer.adapt(sentence_data)
# Convert all sentences to tensors
word_tensors = convert_to_tensor(sentences, dtype=tf.string)
# Use the word tensors to get vectorized phrases
vectorized_words = vectorize_layer(word_tensors)

word_embedding_layer = Embedding(vocab_size, output_length)
embedded_words = word_embedding_layer(vectorized_words)
# 打印输出 / Print output
print(embedded_words)
```

---

➡️ **Next / 下一步**: File 3 of 8

---

### Positional

# 04 — Positional / 04 Positional

**Chapter 14 — File 3 of 8 / 第14章 — 第3个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Positional**.

本脚本演示 **04 Positional**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Embedding

output_length = 6
output_sequence_length = 5

position_embedding_layer = Embedding(output_sequence_length, output_length)
# 生成整数序列 / Generate integer sequence
position_indices = tf.range(output_sequence_length)
embedded_indices = position_embedding_layer(position_indices)
# 打印输出 / Print output
print(embedded_indices)
```

---
## Learning Notes / 学习笔记

- **概念**: Positional 是机器学习中的常用技术。  
  *Positional is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Positional / 04 Positional
# Complete Code / 完整代码
# ===============================

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Embedding

output_length = 6
output_sequence_length = 5

position_embedding_layer = Embedding(output_sequence_length, output_length)
# 生成整数序列 / Generate integer sequence
position_indices = tf.range(output_sequence_length)
embedded_indices = position_embedding_layer(position_indices)
# 打印输出 / Print output
print(embedded_indices)
```

---

➡️ **Next / 下一步**: File 4 of 8

---

### Output

# 05 — Output / 05 Output

**Chapter 14 — File 4 of 8 / 第14章 — 第4个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Create the TextVectorization layer**.

本脚本演示 **Create the TextVectorization layer**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import convert_to_tensor
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import TextVectorization, Embedding
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.data import Dataset

output_length = 6
output_sequence_length = 5
vocab_size = 10
sentences = [["I am a robot"], ["you too robot"]]

sentence_data = Dataset.from_tensor_slices(sentences)
```

---
## Step 2 — Create the TextVectorization layer

```python
vectorize_layer = TextVectorization(output_sequence_length=output_sequence_length,
                                    max_tokens=vocab_size)
```

---
## Step 3 — Train the layer to create a dictionary

```python
vectorize_layer.adapt(sentence_data)
```

---
## Step 4 — Convert all sentences to tensors

```python
word_tensors = convert_to_tensor(sentences, dtype=tf.string)
```

---
## Step 5 — Use the word tensors to get vectorized phrases

```python
vectorized_words = vectorize_layer(word_tensors)

word_embedding_layer = Embedding(vocab_size, output_length)
embedded_words = word_embedding_layer(vectorized_words)

position_embedding_layer = Embedding(output_sequence_length, output_length)
# 生成整数序列 / Generate integer sequence
position_indices = tf.range(output_sequence_length)
embedded_indices = position_embedding_layer(position_indices)

final_output_embedding = embedded_words + embedded_indices
# 打印输出 / Print output
print("Final output: ", final_output_embedding)
```

---
## Learning Notes / 学习笔记

- **概念**: Create the TextVectorization layer 是机器学习中的常用技术。  
  *Create the TextVectorization layer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Output / 05 Output
# Complete Code / 完整代码
# ===============================

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import convert_to_tensor
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import TextVectorization, Embedding
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.data import Dataset

output_length = 6
output_sequence_length = 5
vocab_size = 10
sentences = [["I am a robot"], ["you too robot"]]

sentence_data = Dataset.from_tensor_slices(sentences)
# Create the TextVectorization layer
vectorize_layer = TextVectorization(output_sequence_length=output_sequence_length,
                                    max_tokens=vocab_size)
# Train the layer to create a dictionary
vectorize_layer.adapt(sentence_data)
# Convert all sentences to tensors
word_tensors = convert_to_tensor(sentences, dtype=tf.string)
# Use the word tensors to get vectorized phrases
vectorized_words = vectorize_layer(word_tensors)

word_embedding_layer = Embedding(vocab_size, output_length)
embedded_words = word_embedding_layer(vectorized_words)

position_embedding_layer = Embedding(output_sequence_length, output_length)
# 生成整数序列 / Generate integer sequence
position_indices = tf.range(output_sequence_length)
embedded_indices = position_embedding_layer(position_indices)

final_output_embedding = embedded_words + embedded_indices
# 打印输出 / Print output
print("Final output: ", final_output_embedding)
```

---

➡️ **Next / 下一步**: File 5 of 8

---

### Posembed

# 07 — Posembed / 07 Posembed

**Chapter 14 — File 5 of 8 / 第14章 — 第5个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Create the TextVectorization layer**.

本脚本演示 **Create the TextVectorization layer**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — Step 1

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import convert_to_tensor, string
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import TextVectorization, Embedding, Layer
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.data import Dataset
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

output_length = 6
output_sequence_length = 5
vocab_size = 10
sentences = [["I am a robot"], ["you too robot"]]

sentence_data = Dataset.from_tensor_slices(sentences)
```

---
## Step 2 — Create the TextVectorization layer

```python
vectorize_layer = TextVectorization(output_sequence_length=output_sequence_length,
                                    max_tokens=vocab_size)
```

---
## Step 3 — Train the layer to create a dictionary

```python
vectorize_layer.adapt(sentence_data)
```

---
## Step 4 — Convert all sentences to tensors

```python
word_tensors = convert_to_tensor(sentences, dtype=tf.string)
```

---
## Step 5 — Use the word tensors to get vectorized phrases

```python
vectorized_words = vectorize_layer(word_tensors)

class PositionEmbeddingLayer(Layer):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, seq_length, vocab_size, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.word_embedding_layer = Embedding(
            input_dim=vocab_size, output_dim=output_dim
        )
        self.position_embedding_layer = Embedding(
            input_dim=seq_length, output_dim=output_dim
        )

    def call(self, inputs):
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_words + embedded_indices

my_embedding_layer = PositionEmbeddingLayer(output_sequence_length,
                                            vocab_size, output_length)
embedded_layer_output = my_embedding_layer(vectorized_words)
# 打印输出 / Print output
print("Output from my_embedded_layer: ", embedded_layer_output)
```

---
## Learning Notes / 学习笔记

- **概念**: Create the TextVectorization layer 是机器学习中的常用技术。  
  *Create the TextVectorization layer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Posembed / 07 Posembed
# Complete Code / 完整代码
# ===============================

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import convert_to_tensor, string
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import TextVectorization, Embedding, Layer
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.data import Dataset
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

output_length = 6
output_sequence_length = 5
vocab_size = 10
sentences = [["I am a robot"], ["you too robot"]]

sentence_data = Dataset.from_tensor_slices(sentences)
# Create the TextVectorization layer
vectorize_layer = TextVectorization(output_sequence_length=output_sequence_length,
                                    max_tokens=vocab_size)
# Train the layer to create a dictionary
vectorize_layer.adapt(sentence_data)
# Convert all sentences to tensors
word_tensors = convert_to_tensor(sentences, dtype=tf.string)
# Use the word tensors to get vectorized phrases
vectorized_words = vectorize_layer(word_tensors)

class PositionEmbeddingLayer(Layer):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, seq_length, vocab_size, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.word_embedding_layer = Embedding(
            input_dim=vocab_size, output_dim=output_dim
        )
        self.position_embedding_layer = Embedding(
            input_dim=seq_length, output_dim=output_dim
        )

    def call(self, inputs):
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_words + embedded_indices

my_embedding_layer = PositionEmbeddingLayer(output_sequence_length,
                                            vocab_size, output_length)
embedded_layer_output = my_embedding_layer(vectorized_words)
# 打印输出 / Print output
print("Output from my_embedded_layer: ", embedded_layer_output)
```

---

➡️ **Next / 下一步**: File 6 of 8

---

### Posencoding

# 09 — Posencoding / 09 Posencoding

**Chapter 14 — File 6 of 8 / 第14章 — 第6个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Create the TextVectorization layer**.

本脚本演示 **Create the TextVectorization layer**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — Step 1

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import convert_to_tensor, string
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import TextVectorization, Embedding, Layer
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.data import Dataset
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

output_length = 6
output_sequence_length = 5
vocab_size = 10
sentences = [["I am a robot"], ["you too robot"]]

sentence_data = Dataset.from_tensor_slices(sentences)
```

---
## Step 2 — Create the TextVectorization layer

```python
vectorize_layer = TextVectorization(output_sequence_length=output_sequence_length,
                                    max_tokens=vocab_size)
```

---
## Step 3 — Train the layer to create a dictionary

```python
vectorize_layer.adapt(sentence_data)
```

---
## Step 4 — Convert all sentences to tensors

```python
word_tensors = convert_to_tensor(sentences, dtype=tf.string)
```

---
## Step 5 — Use the word tensors to get vectorized phrases

```python
vectorized_words = vectorize_layer(word_tensors)

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

attnisallyouneed_embedding = PositionEmbeddingFixedWeights(output_sequence_length,
                                            vocab_size, output_length)
attnisallyouneed_output = attnisallyouneed_embedding(vectorized_words)
# 打印输出 / Print output
print("Output from my_embedded_layer: ", attnisallyouneed_output)
```

---
## Learning Notes / 学习笔记

- **概念**: Create the TextVectorization layer 是机器学习中的常用技术。  
  *Create the TextVectorization layer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `matplotlib` | 绑图库 | Plotting library |
| `np.zeros` | 全零数组 | Array filled with zeros |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Posencoding / 09 Posencoding
# Complete Code / 完整代码
# ===============================

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import convert_to_tensor, string
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import TextVectorization, Embedding, Layer
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.data import Dataset
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

output_length = 6
output_sequence_length = 5
vocab_size = 10
sentences = [["I am a robot"], ["you too robot"]]

sentence_data = Dataset.from_tensor_slices(sentences)
# Create the TextVectorization layer
vectorize_layer = TextVectorization(output_sequence_length=output_sequence_length,
                                    max_tokens=vocab_size)
# Train the layer to create a dictionary
vectorize_layer.adapt(sentence_data)
# Convert all sentences to tensors
word_tensors = convert_to_tensor(sentences, dtype=tf.string)
# Use the word tensors to get vectorized phrases
vectorized_words = vectorize_layer(word_tensors)

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

attnisallyouneed_embedding = PositionEmbeddingFixedWeights(output_sequence_length,
                                            vocab_size, output_length)
attnisallyouneed_output = attnisallyouneed_embedding(vectorized_words)
# 打印输出 / Print output
print("Output from my_embedded_layer: ", attnisallyouneed_output)
```

---

➡️ **Next / 下一步**: File 7 of 8

---

### Random



---

### Sinusoidal



---

### Chapter Summary / 章节总结

# Chapter 14 Summary / 第14章总结

## Theme / 主题: Chapter 14 / Chapter 14

This chapter contains **8 code files** demonstrating chapter 14.

本章包含 **8 个代码文件**，演示Chapter 14。

---
## Evolution / 演化路线

  1. `02_vectorize.ipynb` — Vectorize
  2. `03_embedding.ipynb` — Embedding
  3. `04_positional.ipynb` — Positional
  4. `05_output.ipynb` — Output
  5. `07_posembed.ipynb` — Posembed
  6. `09_posencoding.ipynb` — Posencoding
  7. `11_random.ipynb` — Random
  8. `12_sinusoidal.ipynb` — Sinusoidal

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 14) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 14）是机器学习流水线中的基础构建块。

---
