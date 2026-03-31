# LSTM 网络实战 / LSTM Networks with Python
## Lesson 03

---

### Chapter Summary / 章节总结

# Chapter 03 Summary / 第03章总结

## Theme / 主题: Lesson 03 / Lesson 03

This chapter contains **10 code files** demonstrating lesson 03.

本章包含 **10 个代码文件**，演示Lesson 03。

---
## Evolution / 演化路线

  1. `normalize.ipynb` — Normalize
  2. `one_hot_encode.ipynb` — One Hot Encode
  3. `post_seq_padding.ipynb` — Post Seq Padding
  4. `post_seq_truncating.ipynb` — Post Seq Truncating
  5. `pre_seq_padding.ipynb` — Pre Seq Padding
  6. `pre_seq_truncating.ipynb` — Pre Seq Truncating
  7. `shift_backward.ipynb` — Shift Backward
  8. `shift_forward.ipynb` — Shift Forward
  9. `shift_sequence.ipynb` — Shift Sequence
  10. `standardize.ipynb` — Standardize

---
## ML Relevance / ML 关联

The techniques in this chapter (Lesson 03) are fundamental building blocks in machine learning pipelines.

本章技术（Lesson 03）是机器学习流水线中的基础构建块。

---

### Normalize



---

### One Hot Encode



---

### Post Seq Padding

# 01 — Post Seq Padding / Post Seq Padding

**Chapter 03 — File 3 of 10 / 第03章 — 第3个文件（共10个）**

---

## Summary / 总结

This script demonstrates **define sequences**.

本脚本演示 **define sequences**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.sequence import pad_sequences
```

---
## Step 2 — define sequences

```python
sequences = [
	[1, 2, 3, 4],
	   [1, 2, 3],
		     [1]
	]
```

---
## Step 3 — pad sequence

```python
padded = pad_sequences(sequences, padding='post')
# 打印输出 / Print output
print(padded)
```

---
## Learning Notes / 学习笔记

- **概念**: define sequences 是机器学习中的常用技术。  
  *define sequences is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Post Seq Padding / Post Seq Padding
# Complete Code / 完整代码
# ===============================

# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.sequence import pad_sequences
# define sequences
sequences = [
	[1, 2, 3, 4],
	   [1, 2, 3],
		     [1]
	]
# pad sequence
padded = pad_sequences(sequences, padding='post')
# 打印输出 / Print output
print(padded)
```

---

➡️ **Next / 下一步**: File 4 of 10

---

### Post Seq Truncating

# 01 — Post Seq Truncating / Post Seq Truncating

**Chapter 03 — File 4 of 10 / 第03章 — 第4个文件（共10个）**

---

## Summary / 总结

This script demonstrates **define sequences**.

本脚本演示 **define sequences**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.sequence import pad_sequences
```

---
## Step 2 — define sequences

```python
sequences = [
	[1, 2, 3, 4],
	   [1, 2, 3],
		     [1]
	]
```

---
## Step 3 — truncate sequence

```python
truncated= pad_sequences(sequences, maxlen=2, truncating='post')
# 打印输出 / Print output
print(truncated)
```

---
## Learning Notes / 学习笔记

- **概念**: define sequences 是机器学习中的常用技术。  
  *define sequences is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Post Seq Truncating / Post Seq Truncating
# Complete Code / 完整代码
# ===============================

# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.sequence import pad_sequences
# define sequences
sequences = [
	[1, 2, 3, 4],
	   [1, 2, 3],
		     [1]
	]
# truncate sequence
truncated= pad_sequences(sequences, maxlen=2, truncating='post')
# 打印输出 / Print output
print(truncated)
```

---

➡️ **Next / 下一步**: File 5 of 10

---

### Pre Seq Padding



---

### Pre Seq Truncating



---

### Shift Backward

# 01 — Shift Backward / Shift Backward

**Chapter 03 — File 7 of 10 / 第03章 — 第7个文件（共10个）**

---

## Summary / 总结

This script demonstrates **define the sequence**.

本脚本演示 **define the sequence**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 训练模型 / Train the model


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
```

---
## Step 2 — define the sequence

```python
df = DataFrame()
# 生成整数序列 / Generate integer sequence
df['t'] = [x for x in range(10)]
```

---
## Step 3 — shift backward

```python
df['t+1'] = df['t'].shift(-1)
# 打印输出 / Print output
print(df)
```

---
## Learning Notes / 学习笔记

- **概念**: define the sequence 是机器学习中的常用技术。  
  *define the sequence is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Shift Backward / Shift Backward
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# define the sequence
df = DataFrame()
# 生成整数序列 / Generate integer sequence
df['t'] = [x for x in range(10)]
# shift backward
df['t+1'] = df['t'].shift(-1)
# 打印输出 / Print output
print(df)
```

---

➡️ **Next / 下一步**: File 8 of 10

---

### Shift Forward

# 01 — Shift Forward / Shift Forward

**Chapter 03 — File 8 of 10 / 第03章 — 第8个文件（共10个）**

---

## Summary / 总结

This script demonstrates **define the sequence**.

本脚本演示 **define the sequence**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
```

---
## Step 2 — define the sequence

```python
df = DataFrame()
# 生成整数序列 / Generate integer sequence
df['t'] = [x for x in range(10)]
```

---
## Step 3 — shift forward

```python
df['t-1'] = df['t'].shift(1)
# 打印输出 / Print output
print(df)
```

---
## Learning Notes / 学习笔记

- **概念**: define the sequence 是机器学习中的常用技术。  
  *define the sequence is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Shift Forward / Shift Forward
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# define the sequence
df = DataFrame()
# 生成整数序列 / Generate integer sequence
df['t'] = [x for x in range(10)]
# shift forward
df['t-1'] = df['t'].shift(1)
# 打印输出 / Print output
print(df)
```

---

➡️ **Next / 下一步**: File 9 of 10

---

### Shift Sequence

# 01 — Shift Sequence / Shift Sequence

**Chapter 03 — File 9 of 10 / 第03章 — 第9个文件（共10个）**

---

## Summary / 总结

This script demonstrates **define the sequence**.

本脚本演示 **define the sequence**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
```

---
## Step 2 — define the sequence

```python
df = DataFrame()
# 生成整数序列 / Generate integer sequence
df['t'] = [x for x in range(10)]
# 打印输出 / Print output
print(df)
```

---
## Learning Notes / 学习笔记

- **概念**: define the sequence 是机器学习中的常用技术。  
  *define the sequence is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Shift Sequence / Shift Sequence
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# define the sequence
df = DataFrame()
# 生成整数序列 / Generate integer sequence
df['t'] = [x for x in range(10)]
# 打印输出 / Print output
print(df)
```

---

➡️ **Next / 下一步**: File 10 of 10

---

### Standardize



---
