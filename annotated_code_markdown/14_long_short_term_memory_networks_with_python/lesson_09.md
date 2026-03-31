# LSTM 网络实战 / LSTM Networks with Python
## Lesson 09

---

### Chapter Summary / 章节总结



---

### Encoder Decoder Lstm



---

### Problem Generate Pairs

# 01 — Problem Generate Pairs / Problem Generate Pairs

**Chapter 09 — File 2 of 5 / 第09章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **generate lists of random integers and their sum**.

本脚本演示 **generate lists of random integers and their sum**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from random import seed
from random import randint
```

---
## Step 2 — generate lists of random integers and their sum

```python
def random_sum_pairs(n_examples, n_numbers, largest):
	X, y = list(), list()
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_examples):
  # 生成整数序列 / Generate integer sequence
		in_pattern = [randint(1,largest) for _ in range(n_numbers)]
		out_pattern = sum(in_pattern)
  # 添加元素到列表末尾 / Append element to list end
		X.append(in_pattern)
  # 添加元素到列表末尾 / Append element to list end
		y.append(out_pattern)
	return X, y

# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
n_samples = 1
n_numbers = 2
largest = 10
```

---
## Step 3 — generate pairs

```python
X, y = random_sum_pairs(n_samples, n_numbers, largest)
# 打印输出 / Print output
print(X, y)
```

---
## Learning Notes / 学习笔记

- **概念**: generate lists of random integers and their sum 是机器学习中的常用技术。  
  *generate lists of random integers and their sum is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Problem Generate Pairs / Problem Generate Pairs
# Complete Code / 完整代码
# ===============================

from random import seed
from random import randint

# generate lists of random integers and their sum
def random_sum_pairs(n_examples, n_numbers, largest):
	X, y = list(), list()
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_examples):
  # 生成整数序列 / Generate integer sequence
		in_pattern = [randint(1,largest) for _ in range(n_numbers)]
		out_pattern = sum(in_pattern)
  # 添加元素到列表末尾 / Append element to list end
		X.append(in_pattern)
  # 添加元素到列表末尾 / Append element to list end
		y.append(out_pattern)
	return X, y

# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
n_samples = 1
n_numbers = 2
largest = 10
# generate pairs
X, y = random_sum_pairs(n_samples, n_numbers, largest)
# 打印输出 / Print output
print(X, y)
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Problem Integer Encode

# 01 — Problem Integer Encode / 数据编码

**Chapter 09 — File 3 of 5 / 第09章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **generate lists of random integers and their sum**.

本脚本演示 **generate lists of random integers and their sum**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from random import seed
from random import randint
from math import ceil
from math import log10
```

---
## Step 2 — generate lists of random integers and their sum

```python
def random_sum_pairs(n_examples, n_numbers, largest):
	X, y = list(), list()
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_examples):
  # 生成整数序列 / Generate integer sequence
		in_pattern = [randint(1,largest) for _ in range(n_numbers)]
		out_pattern = sum(in_pattern)
  # 添加元素到列表末尾 / Append element to list end
		X.append(in_pattern)
  # 添加元素到列表末尾 / Append element to list end
		y.append(out_pattern)
	return X, y
```

---
## Step 3 — convert data to strings

```python
def to_string(X, y, n_numbers, largest):
	max_length = int(n_numbers * ceil(log10(largest+1)) + n_numbers - 1)
	Xstr = list()
	for pattern in X:
		strp = '+'.join([str(n) for n in pattern])
  # 获取长度 / Get length
		strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
  # 添加元素到列表末尾 / Append element to list end
		Xstr.append(strp)
	max_length = int(ceil(log10(n_numbers * (largest+1))))
	ystr = list()
	for pattern in y:
		strp = str(pattern)
  # 获取长度 / Get length
		strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
  # 添加元素到列表末尾 / Append element to list end
		ystr.append(strp)
	return Xstr, ystr
```

---
## Step 4 — integer encode strings

```python
def integer_encode(X, y, alphabet):
 # 同时获取索引和值 / Get both index and value
	char_to_int = dict((c, i) for i, c in enumerate(alphabet))
	Xenc = list()
	for pattern in X:
		integer_encoded = [char_to_int[char] for char in pattern]
  # 添加元素到列表末尾 / Append element to list end
		Xenc.append(integer_encoded)
	yenc = list()
	for pattern in y:
		integer_encoded = [char_to_int[char] for char in pattern]
  # 添加元素到列表末尾 / Append element to list end
		yenc.append(integer_encoded)
	return Xenc, yenc

# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
n_samples = 1
n_numbers = 2
largest = 10
```

---
## Step 5 — generate pairs

```python
X, y = random_sum_pairs(n_samples, n_numbers, largest)
# 打印输出 / Print output
print(X, y)
```

---
## Step 6 — convert to strings

```python
X, y = to_string(X, y, n_numbers, largest)
# 打印输出 / Print output
print(X, y)
```

---
## Step 7 — integer encode

```python
alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ']
X, y = integer_encode(X, y, alphabet)
# 打印输出 / Print output
print(X, y)
```

---
## Learning Notes / 学习笔记

- **概念**: generate lists of random integers and their sum 是机器学习中的常用技术。  
  *generate lists of random integers and their sum is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Problem Integer Encode / 数据编码
# Complete Code / 完整代码
# ===============================

from random import seed
from random import randint
from math import ceil
from math import log10

# generate lists of random integers and their sum
def random_sum_pairs(n_examples, n_numbers, largest):
	X, y = list(), list()
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_examples):
  # 生成整数序列 / Generate integer sequence
		in_pattern = [randint(1,largest) for _ in range(n_numbers)]
		out_pattern = sum(in_pattern)
  # 添加元素到列表末尾 / Append element to list end
		X.append(in_pattern)
  # 添加元素到列表末尾 / Append element to list end
		y.append(out_pattern)
	return X, y

# convert data to strings
def to_string(X, y, n_numbers, largest):
	max_length = int(n_numbers * ceil(log10(largest+1)) + n_numbers - 1)
	Xstr = list()
	for pattern in X:
		strp = '+'.join([str(n) for n in pattern])
  # 获取长度 / Get length
		strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
  # 添加元素到列表末尾 / Append element to list end
		Xstr.append(strp)
	max_length = int(ceil(log10(n_numbers * (largest+1))))
	ystr = list()
	for pattern in y:
		strp = str(pattern)
  # 获取长度 / Get length
		strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
  # 添加元素到列表末尾 / Append element to list end
		ystr.append(strp)
	return Xstr, ystr

# integer encode strings
def integer_encode(X, y, alphabet):
 # 同时获取索引和值 / Get both index and value
	char_to_int = dict((c, i) for i, c in enumerate(alphabet))
	Xenc = list()
	for pattern in X:
		integer_encoded = [char_to_int[char] for char in pattern]
  # 添加元素到列表末尾 / Append element to list end
		Xenc.append(integer_encoded)
	yenc = list()
	for pattern in y:
		integer_encoded = [char_to_int[char] for char in pattern]
  # 添加元素到列表末尾 / Append element to list end
		yenc.append(integer_encoded)
	return Xenc, yenc

# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
n_samples = 1
n_numbers = 2
largest = 10
# generate pairs
X, y = random_sum_pairs(n_samples, n_numbers, largest)
# 打印输出 / Print output
print(X, y)
# convert to strings
X, y = to_string(X, y, n_numbers, largest)
# 打印输出 / Print output
print(X, y)
# integer encode
alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ']
X, y = integer_encode(X, y, alphabet)
# 打印输出 / Print output
print(X, y)
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Problem One Hot

# 01 — Problem One Hot / Problem One Hot

**Chapter 09 — File 4 of 5 / 第09章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **generate lists of random integers and their sum**.

本脚本演示 **generate lists of random integers and their sum**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from random import seed
from random import randint
from math import ceil
from math import log10
```

---
## Step 2 — generate lists of random integers and their sum

```python
def random_sum_pairs(n_examples, n_numbers, largest):
	X, y = list(), list()
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_examples):
  # 生成整数序列 / Generate integer sequence
		in_pattern = [randint(1,largest) for _ in range(n_numbers)]
		out_pattern = sum(in_pattern)
  # 添加元素到列表末尾 / Append element to list end
		X.append(in_pattern)
  # 添加元素到列表末尾 / Append element to list end
		y.append(out_pattern)
	return X, y
```

---
## Step 3 — convert data to strings

```python
def to_string(X, y, n_numbers, largest):
	max_length = int(n_numbers * ceil(log10(largest+1)) + n_numbers - 1)
	Xstr = list()
	for pattern in X:
		strp = '+'.join([str(n) for n in pattern])
  # 获取长度 / Get length
		strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
  # 添加元素到列表末尾 / Append element to list end
		Xstr.append(strp)
	max_length = int(ceil(log10(n_numbers * (largest+1))))
	ystr = list()
	for pattern in y:
		strp = str(pattern)
  # 获取长度 / Get length
		strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
  # 添加元素到列表末尾 / Append element to list end
		ystr.append(strp)
	return Xstr, ystr
```

---
## Step 4 — integer encode strings

```python
def integer_encode(X, y, alphabet):
 # 同时获取索引和值 / Get both index and value
	char_to_int = dict((c, i) for i, c in enumerate(alphabet))
	Xenc = list()
	for pattern in X:
		integer_encoded = [char_to_int[char] for char in pattern]
  # 添加元素到列表末尾 / Append element to list end
		Xenc.append(integer_encoded)
	yenc = list()
	for pattern in y:
		integer_encoded = [char_to_int[char] for char in pattern]
  # 添加元素到列表末尾 / Append element to list end
		yenc.append(integer_encoded)
	return Xenc, yenc
```

---
## Step 5 — one hot encode

```python
def one_hot_encode(X, y, max_int):
	Xenc = list()
	for seq in X:
		pattern = list()
		for index in seq:
   # 生成整数序列 / Generate integer sequence
			vector = [0 for _ in range(max_int)]
			vector[index] = 1
   # 添加元素到列表末尾 / Append element to list end
			pattern.append(vector)
  # 添加元素到列表末尾 / Append element to list end
		Xenc.append(pattern)
	yenc = list()
	for seq in y:
		pattern = list()
		for index in seq:
   # 生成整数序列 / Generate integer sequence
			vector = [0 for _ in range(max_int)]
			vector[index] = 1
   # 添加元素到列表末尾 / Append element to list end
			pattern.append(vector)
  # 添加元素到列表末尾 / Append element to list end
		yenc.append(pattern)
	return Xenc, yenc

# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
n_samples = 1
n_numbers = 2
largest = 10
```

---
## Step 6 — generate pairs

```python
X, y = random_sum_pairs(n_samples, n_numbers, largest)
# 打印输出 / Print output
print(X, y)
```

---
## Step 7 — convert to strings

```python
X, y = to_string(X, y, n_numbers, largest)
# 打印输出 / Print output
print(X, y)
```

---
## Step 8 — integer encode

```python
alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ']
X, y = integer_encode(X, y, alphabet)
# 打印输出 / Print output
print(X, y)
```

---
## Step 9 — one hot encode

```python
# 获取长度 / Get length
X, y = one_hot_encode(X, y, len(alphabet))
# 打印输出 / Print output
print(X, y)
```

---
## Learning Notes / 学习笔记

- **概念**: generate lists of random integers and their sum 是机器学习中的常用技术。  
  *generate lists of random integers and their sum is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Problem One Hot / Problem One Hot
# Complete Code / 完整代码
# ===============================

from random import seed
from random import randint
from math import ceil
from math import log10

# generate lists of random integers and their sum
def random_sum_pairs(n_examples, n_numbers, largest):
	X, y = list(), list()
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_examples):
  # 生成整数序列 / Generate integer sequence
		in_pattern = [randint(1,largest) for _ in range(n_numbers)]
		out_pattern = sum(in_pattern)
  # 添加元素到列表末尾 / Append element to list end
		X.append(in_pattern)
  # 添加元素到列表末尾 / Append element to list end
		y.append(out_pattern)
	return X, y

# convert data to strings
def to_string(X, y, n_numbers, largest):
	max_length = int(n_numbers * ceil(log10(largest+1)) + n_numbers - 1)
	Xstr = list()
	for pattern in X:
		strp = '+'.join([str(n) for n in pattern])
  # 获取长度 / Get length
		strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
  # 添加元素到列表末尾 / Append element to list end
		Xstr.append(strp)
	max_length = int(ceil(log10(n_numbers * (largest+1))))
	ystr = list()
	for pattern in y:
		strp = str(pattern)
  # 获取长度 / Get length
		strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
  # 添加元素到列表末尾 / Append element to list end
		ystr.append(strp)
	return Xstr, ystr

# integer encode strings
def integer_encode(X, y, alphabet):
 # 同时获取索引和值 / Get both index and value
	char_to_int = dict((c, i) for i, c in enumerate(alphabet))
	Xenc = list()
	for pattern in X:
		integer_encoded = [char_to_int[char] for char in pattern]
  # 添加元素到列表末尾 / Append element to list end
		Xenc.append(integer_encoded)
	yenc = list()
	for pattern in y:
		integer_encoded = [char_to_int[char] for char in pattern]
  # 添加元素到列表末尾 / Append element to list end
		yenc.append(integer_encoded)
	return Xenc, yenc

# one hot encode
def one_hot_encode(X, y, max_int):
	Xenc = list()
	for seq in X:
		pattern = list()
		for index in seq:
   # 生成整数序列 / Generate integer sequence
			vector = [0 for _ in range(max_int)]
			vector[index] = 1
   # 添加元素到列表末尾 / Append element to list end
			pattern.append(vector)
  # 添加元素到列表末尾 / Append element to list end
		Xenc.append(pattern)
	yenc = list()
	for seq in y:
		pattern = list()
		for index in seq:
   # 生成整数序列 / Generate integer sequence
			vector = [0 for _ in range(max_int)]
			vector[index] = 1
   # 添加元素到列表末尾 / Append element to list end
			pattern.append(vector)
  # 添加元素到列表末尾 / Append element to list end
		yenc.append(pattern)
	return Xenc, yenc

# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
n_samples = 1
n_numbers = 2
largest = 10
# generate pairs
X, y = random_sum_pairs(n_samples, n_numbers, largest)
# 打印输出 / Print output
print(X, y)
# convert to strings
X, y = to_string(X, y, n_numbers, largest)
# 打印输出 / Print output
print(X, y)
# integer encode
alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ']
X, y = integer_encode(X, y, alphabet)
# 打印输出 / Print output
print(X, y)
# one hot encode
# 获取长度 / Get length
X, y = one_hot_encode(X, y, len(alphabet))
# 打印输出 / Print output
print(X, y)
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Problem Pad



---
