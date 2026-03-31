# 不平衡分类问题 / Imbalanced Classification with Python
## Chapter 08

---

### Dataset



---

### Logloss

# 02 — Logloss / 损失函数

**Chapter 08 — File 2 of 4 / 第08章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **log loss for naive probability predictions.**.

本脚本演示 **log loss for naive probability predictions.**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  ✂️ 划分数据集 / Split Dataset
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  ⚙️ 配置训练 / Configure Training
```

---
## Step 1 — log loss for naive probability predictions.

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import log_loss
```

---
## Step 2 — generate 2 class dataset

```python
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99], flip_y=0, random_state=1)
```

---
## Step 3 — split into train/test sets with same class ratio

```python
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
```

---
## Step 4 — no skill prediction 0

```python
# 获取长度 / Get length
probabilities = [[1, 0] for _ in range(len(testy))]
avg_logloss = log_loss(testy, probabilities)
# 打印输出 / Print output
print('P(class0=1): Log Loss=%.3f' % (avg_logloss))
```

---
## Step 5 — no skill prediction 1

```python
# 获取长度 / Get length
probabilities = [[0, 1] for _ in range(len(testy))]
avg_logloss = log_loss(testy, probabilities)
# 打印输出 / Print output
print('P(class1=1): Log Loss=%.3f' % (avg_logloss))
```

---
## Step 6 — baseline probabilities

```python
# 获取长度 / Get length
probabilities = [[0.99, 0.01] for _ in range(len(testy))]
avg_logloss = log_loss(testy, probabilities)
# 打印输出 / Print output
print('Baseline: Log Loss=%.3f' % (avg_logloss))
```

---
## Step 7 — perfect probabilities

```python
avg_logloss = log_loss(testy, testy)
# 打印输出 / Print output
print('Perfect: Log Loss=%.3f' % (avg_logloss))
```

---
## Learning Notes / 学习笔记

- **概念**: log loss for naive probability predictions. 是机器学习中的常用技术。  
  *log loss for naive probability predictions. is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Logloss / 损失函数
# Complete Code / 完整代码
# ===============================

# log loss for naive probability predictions.
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import log_loss
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99], flip_y=0, random_state=1)
# split into train/test sets with same class ratio
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# no skill prediction 0
# 获取长度 / Get length
probabilities = [[1, 0] for _ in range(len(testy))]
avg_logloss = log_loss(testy, probabilities)
# 打印输出 / Print output
print('P(class0=1): Log Loss=%.3f' % (avg_logloss))
# no skill prediction 1
# 获取长度 / Get length
probabilities = [[0, 1] for _ in range(len(testy))]
avg_logloss = log_loss(testy, probabilities)
# 打印输出 / Print output
print('P(class1=1): Log Loss=%.3f' % (avg_logloss))
# baseline probabilities
# 获取长度 / Get length
probabilities = [[0.99, 0.01] for _ in range(len(testy))]
avg_logloss = log_loss(testy, probabilities)
# 打印输出 / Print output
print('Baseline: Log Loss=%.3f' % (avg_logloss))
# perfect probabilities
avg_logloss = log_loss(testy, testy)
# 打印输出 / Print output
print('Perfect: Log Loss=%.3f' % (avg_logloss))
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Brier

# 03 — Brier / 03 Brier

**Chapter 08 — File 3 of 4 / 第08章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **brier score for naive probability predictions.**.

本脚本演示 **brier score for naive probability predictions.**。

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
  ✂️ 划分数据集 / Split Dataset
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — brier score for naive probability predictions.

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import brier_score_loss
```

---
## Step 2 — generate 2 class dataset

```python
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99], flip_y=0, random_state=1)
```

---
## Step 3 — split into train/test sets with same class ratio

```python
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
```

---
## Step 4 — no skill prediction 0

```python
# 获取长度 / Get length
probabilities = [0.0 for _ in range(len(testy))]
avg_brier = brier_score_loss(testy, probabilities)
# 打印输出 / Print output
print('P(class1=0): Brier Score=%.4f' % (avg_brier))
```

---
## Step 5 — no skill prediction 1

```python
# 获取长度 / Get length
probabilities = [1.0 for _ in range(len(testy))]
avg_brier = brier_score_loss(testy, probabilities)
# 打印输出 / Print output
print('P(class1=1): Brier Score=%.4f' % (avg_brier))
```

---
## Step 6 — baseline probabilities

```python
# 获取长度 / Get length
probabilities = [0.01 for _ in range(len(testy))]
avg_brier = brier_score_loss(testy, probabilities)
# 打印输出 / Print output
print('Baseline: Brier Score=%.4f' % (avg_brier))
```

---
## Step 7 — perfect probabilities

```python
avg_brier = brier_score_loss(testy, testy)
# 打印输出 / Print output
print('Perfect: Brier Score=%.4f' % (avg_brier))
```

---
## Learning Notes / 学习笔记

- **概念**: brier score for naive probability predictions. 是机器学习中的常用技术。  
  *brier score for naive probability predictions. is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Brier / 03 Brier
# Complete Code / 完整代码
# ===============================

# brier score for naive probability predictions.
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import brier_score_loss
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99], flip_y=0, random_state=1)
# split into train/test sets with same class ratio
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# no skill prediction 0
# 获取长度 / Get length
probabilities = [0.0 for _ in range(len(testy))]
avg_brier = brier_score_loss(testy, probabilities)
# 打印输出 / Print output
print('P(class1=0): Brier Score=%.4f' % (avg_brier))
# no skill prediction 1
# 获取长度 / Get length
probabilities = [1.0 for _ in range(len(testy))]
avg_brier = brier_score_loss(testy, probabilities)
# 打印输出 / Print output
print('P(class1=1): Brier Score=%.4f' % (avg_brier))
# baseline probabilities
# 获取长度 / Get length
probabilities = [0.01 for _ in range(len(testy))]
avg_brier = brier_score_loss(testy, probabilities)
# 打印输出 / Print output
print('Baseline: Brier Score=%.4f' % (avg_brier))
# perfect probabilities
avg_brier = brier_score_loss(testy, testy)
# 打印输出 / Print output
print('Perfect: Brier Score=%.4f' % (avg_brier))
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Brier Skill Score

# 04 — Brier Skill Score / 04 Brier Skill Score

**Chapter 08 — File 4 of 4 / 第08章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **brier skill score for naive probability predictions.**.

本脚本演示 **brier skill score for naive probability predictions.**。

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
  ✂️ 划分数据集 / Split Dataset
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — brier skill score for naive probability predictions.

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import brier_score_loss
```

---
## Step 2 — calculate the brier skill score

```python
def brier_skill_score(y, yhat, brier_ref):
```

---
## Step 3 — calculate the brier score

```python
bs = brier_score_loss(y, yhat)
```

---
## Step 4 — calculate skill score

```python
return 1.0 - (bs / brier_ref)
```

---
## Step 5 — generate 2 class dataset

```python
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99], flip_y=0, random_state=1)
```

---
## Step 6 — split into train/test sets with same class ratio

```python
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
```

---
## Step 7 — calculate reference

```python
# 获取长度 / Get length
probabilities = [0.01 for _ in range(len(testy))]
brier_ref = brier_score_loss(testy, probabilities)
# 打印输出 / Print output
print('Reference: Brier Score=%.4f' % (brier_ref))
```

---
## Step 8 — no skill prediction 0

```python
# 获取长度 / Get length
probabilities = [0.0 for _ in range(len(testy))]
bss = brier_skill_score(testy, probabilities, brier_ref)
# 打印输出 / Print output
print('P(class1=0): BSS=%.4f' % (bss))
```

---
## Step 9 — no skill prediction 1

```python
# 获取长度 / Get length
probabilities = [1.0 for _ in range(len(testy))]
bss = brier_skill_score(testy, probabilities, brier_ref)
# 打印输出 / Print output
print('P(class1=1): BSS=%.4f' % (bss))
```

---
## Step 10 — baseline probabilities

```python
# 获取长度 / Get length
probabilities = [0.01 for _ in range(len(testy))]
bss = brier_skill_score(testy, probabilities, brier_ref)
# 打印输出 / Print output
print('Baseline: BSS=%.4f' % (bss))
```

---
## Step 11 — perfect probabilities

```python
bss = brier_skill_score(testy, testy, brier_ref)
# 打印输出 / Print output
print('Perfect: BSS=%.4f' % (bss))
```

---
## Learning Notes / 学习笔记

- **概念**: brier skill score for naive probability predictions. 是机器学习中的常用技术。  
  *brier skill score for naive probability predictions. is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Brier Skill Score / 04 Brier Skill Score
# Complete Code / 完整代码
# ===============================

# brier skill score for naive probability predictions.
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import brier_score_loss

# calculate the brier skill score
def brier_skill_score(y, yhat, brier_ref):
	# calculate the brier score
	bs = brier_score_loss(y, yhat)
	# calculate skill score
	return 1.0 - (bs / brier_ref)

# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99], flip_y=0, random_state=1)
# split into train/test sets with same class ratio
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# calculate reference
# 获取长度 / Get length
probabilities = [0.01 for _ in range(len(testy))]
brier_ref = brier_score_loss(testy, probabilities)
# 打印输出 / Print output
print('Reference: Brier Score=%.4f' % (brier_ref))
# no skill prediction 0
# 获取长度 / Get length
probabilities = [0.0 for _ in range(len(testy))]
bss = brier_skill_score(testy, probabilities, brier_ref)
# 打印输出 / Print output
print('P(class1=0): BSS=%.4f' % (bss))
# no skill prediction 1
# 获取长度 / Get length
probabilities = [1.0 for _ in range(len(testy))]
bss = brier_skill_score(testy, probabilities, brier_ref)
# 打印输出 / Print output
print('P(class1=1): BSS=%.4f' % (bss))
# baseline probabilities
# 获取长度 / Get length
probabilities = [0.01 for _ in range(len(testy))]
bss = brier_skill_score(testy, probabilities, brier_ref)
# 打印输出 / Print output
print('Baseline: BSS=%.4f' % (bss))
# perfect probabilities
bss = brier_skill_score(testy, testy, brier_ref)
# 打印输出 / Print output
print('Perfect: BSS=%.4f' % (bss))
```

---

### Chapter Summary / 章节总结



---
