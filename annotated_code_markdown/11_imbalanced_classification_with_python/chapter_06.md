# 不平衡分类问题 / Imbalanced Classification with Python
## Chapter 06

---

### Precision Binary

# 01 — Precision Binary / 01 Precision Binary

**Chapter 06 — File 1 of 5 / 第06章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **calculates precision for 1:100 dataset with 90 tp and 30 fp**.

本脚本演示 **calculates precision for 1:100 dataset with 90 tp and 30 fp**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — calculates precision for 1:100 dataset with 90 tp and 30 fp

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import precision_score
```

---
## Step 2 — define actual

```python
# 生成整数序列 / Generate integer sequence
act_pos = [1 for _ in range(100)]
# 生成整数序列 / Generate integer sequence
act_neg = [0 for _ in range(10000)]
y_true = act_pos + act_neg
```

---
## Step 3 — define predictions

```python
# 生成整数序列 / Generate integer sequence
pred_pos = [0 for _ in range(10)] + [1 for _ in range(90)]
# 生成整数序列 / Generate integer sequence
pred_neg = [1 for _ in range(30)] + [0 for _ in range(9970)]
y_pred = pred_pos + pred_neg
```

---
## Step 4 — calculate prediction

```python
# 计算精确率 = TP / (TP+FP) / Precision = TP / (TP+FP)
precision = precision_score(y_true, y_pred, average='binary')
# 打印输出 / Print output
print('Precision: %.3f' % precision)
```

---
## Learning Notes / 学习笔记

- **概念**: calculates precision for 1:100 dataset with 90 tp and 30 fp 是机器学习中的常用技术。  
  *calculates precision for 1:100 dataset with 90 tp and 30 fp is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Precision Binary / 01 Precision Binary
# Complete Code / 完整代码
# ===============================

# calculates precision for 1:100 dataset with 90 tp and 30 fp
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import precision_score
# define actual
# 生成整数序列 / Generate integer sequence
act_pos = [1 for _ in range(100)]
# 生成整数序列 / Generate integer sequence
act_neg = [0 for _ in range(10000)]
y_true = act_pos + act_neg
# define predictions
# 生成整数序列 / Generate integer sequence
pred_pos = [0 for _ in range(10)] + [1 for _ in range(90)]
# 生成整数序列 / Generate integer sequence
pred_neg = [1 for _ in range(30)] + [0 for _ in range(9970)]
y_pred = pred_pos + pred_neg
# calculate prediction
# 计算精确率 = TP / (TP+FP) / Precision = TP / (TP+FP)
precision = precision_score(y_true, y_pred, average='binary')
# 打印输出 / Print output
print('Precision: %.3f' % precision)
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Precision Multiclass

# 02 — Precision Multiclass / 02 Precision Multiclass

**Chapter 06 — File 2 of 5 / 第06章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **calculates precision for 1:1:100 dataset with 50tp,20fp, 99tp,51fp**.

本脚本演示 **calculates precision for 1:1:100 dataset with 50tp,20fp, 99tp,51fp**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — calculates precision for 1:1:100 dataset with 50tp,20fp, 99tp,51fp

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import precision_score
```

---
## Step 2 — define actual

```python
# 生成整数序列 / Generate integer sequence
act_pos1 = [1 for _ in range(100)]
# 生成整数序列 / Generate integer sequence
act_pos2 = [2 for _ in range(100)]
# 生成整数序列 / Generate integer sequence
act_neg = [0 for _ in range(10000)]
y_true = act_pos1 + act_pos2 + act_neg
```

---
## Step 3 — define predictions

```python
# 生成整数序列 / Generate integer sequence
pred_pos1 = [0 for _ in range(50)] + [1 for _ in range(50)]
# 生成整数序列 / Generate integer sequence
pred_pos2 = [0 for _ in range(1)] + [2 for _ in range(99)]
# 生成整数序列 / Generate integer sequence
pred_neg = [1 for _ in range(20)] + [2 for _ in range(51)] + [0 for _ in range(9929)]
y_pred = pred_pos1 + pred_pos2 + pred_neg
```

---
## Step 4 — calculate prediction

```python
# 计算精确率 = TP / (TP+FP) / Precision = TP / (TP+FP)
precision = precision_score(y_true, y_pred, labels=[1,2], average='micro')
# 打印输出 / Print output
print('Precision: %.3f' % precision)
```

---
## Learning Notes / 学习笔记

- **概念**: calculates precision for 1:1:100 dataset with 50tp,20fp, 99tp,51fp 是机器学习中的常用技术。  
  *calculates precision for 1:1:100 dataset with 50tp,20fp, 99tp,51fp is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Precision Multiclass / 02 Precision Multiclass
# Complete Code / 完整代码
# ===============================

# calculates precision for 1:1:100 dataset with 50tp,20fp, 99tp,51fp
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import precision_score
# define actual
# 生成整数序列 / Generate integer sequence
act_pos1 = [1 for _ in range(100)]
# 生成整数序列 / Generate integer sequence
act_pos2 = [2 for _ in range(100)]
# 生成整数序列 / Generate integer sequence
act_neg = [0 for _ in range(10000)]
y_true = act_pos1 + act_pos2 + act_neg
# define predictions
# 生成整数序列 / Generate integer sequence
pred_pos1 = [0 for _ in range(50)] + [1 for _ in range(50)]
# 生成整数序列 / Generate integer sequence
pred_pos2 = [0 for _ in range(1)] + [2 for _ in range(99)]
# 生成整数序列 / Generate integer sequence
pred_neg = [1 for _ in range(20)] + [2 for _ in range(51)] + [0 for _ in range(9929)]
y_pred = pred_pos1 + pred_pos2 + pred_neg
# calculate prediction
# 计算精确率 = TP / (TP+FP) / Precision = TP / (TP+FP)
precision = precision_score(y_true, y_pred, labels=[1,2], average='micro')
# 打印输出 / Print output
print('Precision: %.3f' % precision)
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Recall Binary

# 03 — Recall Binary / 03 Recall Binary

**Chapter 06 — File 3 of 5 / 第06章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **calculates recall for 1:100 dataset with 90 tp and 10 fn**.

本脚本演示 **calculates recall for 1:100 dataset with 90 tp and 10 fn**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — calculates recall for 1:100 dataset with 90 tp and 10 fn

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import recall_score
```

---
## Step 2 — define actual

```python
# 生成整数序列 / Generate integer sequence
act_pos = [1 for _ in range(100)]
# 生成整数序列 / Generate integer sequence
act_neg = [0 for _ in range(10000)]
y_true = act_pos + act_neg
```

---
## Step 3 — define predictions

```python
# 生成整数序列 / Generate integer sequence
pred_pos = [0 for _ in range(10)] + [1 for _ in range(90)]
# 生成整数序列 / Generate integer sequence
pred_neg = [0 for _ in range(10000)]
y_pred = pred_pos + pred_neg
```

---
## Step 4 — calculate recall

```python
# 计算召回率 = TP / (TP+FN) / Recall = TP / (TP+FN)
recall = recall_score(y_true, y_pred, average='binary')
# 打印输出 / Print output
print('Recall: %.3f' % recall)
```

---
## Learning Notes / 学习笔记

- **概念**: calculates recall for 1:100 dataset with 90 tp and 10 fn 是机器学习中的常用技术。  
  *calculates recall for 1:100 dataset with 90 tp and 10 fn is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Recall Binary / 03 Recall Binary
# Complete Code / 完整代码
# ===============================

# calculates recall for 1:100 dataset with 90 tp and 10 fn
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import recall_score
# define actual
# 生成整数序列 / Generate integer sequence
act_pos = [1 for _ in range(100)]
# 生成整数序列 / Generate integer sequence
act_neg = [0 for _ in range(10000)]
y_true = act_pos + act_neg
# define predictions
# 生成整数序列 / Generate integer sequence
pred_pos = [0 for _ in range(10)] + [1 for _ in range(90)]
# 生成整数序列 / Generate integer sequence
pred_neg = [0 for _ in range(10000)]
y_pred = pred_pos + pred_neg
# calculate recall
# 计算召回率 = TP / (TP+FN) / Recall = TP / (TP+FN)
recall = recall_score(y_true, y_pred, average='binary')
# 打印输出 / Print output
print('Recall: %.3f' % recall)
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Recall Multiclass

# 04 — Recall Multiclass / 04 Recall Multiclass

**Chapter 06 — File 4 of 5 / 第06章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **calculates recall for 1:1:100 dataset with 77tp,23fn and 95tp,5fn**.

本脚本演示 **calculates recall for 1:1:100 dataset with 77tp,23fn and 95tp,5fn**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — calculates recall for 1:1:100 dataset with 77tp,23fn and 95tp,5fn

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import recall_score
```

---
## Step 2 — define actual

```python
# 生成整数序列 / Generate integer sequence
act_pos1 = [1 for _ in range(100)]
# 生成整数序列 / Generate integer sequence
act_pos2 = [2 for _ in range(100)]
# 生成整数序列 / Generate integer sequence
act_neg = [0 for _ in range(10000)]
y_true = act_pos1 + act_pos2 + act_neg
```

---
## Step 3 — define predictions

```python
# 生成整数序列 / Generate integer sequence
pred_pos1 = [0 for _ in range(23)] + [1 for _ in range(77)]
# 生成整数序列 / Generate integer sequence
pred_pos2 = [0 for _ in range(5)] + [2 for _ in range(95)]
# 生成整数序列 / Generate integer sequence
pred_neg = [0 for _ in range(10000)]
y_pred = pred_pos1 + pred_pos2 + pred_neg
```

---
## Step 4 — calculate recall

```python
# 计算召回率 = TP / (TP+FN) / Recall = TP / (TP+FN)
recall = recall_score(y_true, y_pred, labels=[1,2], average='micro')
# 打印输出 / Print output
print('Recall: %.3f' % recall)
```

---
## Learning Notes / 学习笔记

- **概念**: calculates recall for 1:1:100 dataset with 77tp,23fn and 95tp,5fn 是机器学习中的常用技术。  
  *calculates recall for 1:1:100 dataset with 77tp,23fn and 95tp,5fn is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Recall Multiclass / 04 Recall Multiclass
# Complete Code / 完整代码
# ===============================

# calculates recall for 1:1:100 dataset with 77tp,23fn and 95tp,5fn
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import recall_score
# define actual
# 生成整数序列 / Generate integer sequence
act_pos1 = [1 for _ in range(100)]
# 生成整数序列 / Generate integer sequence
act_pos2 = [2 for _ in range(100)]
# 生成整数序列 / Generate integer sequence
act_neg = [0 for _ in range(10000)]
y_true = act_pos1 + act_pos2 + act_neg
# define predictions
# 生成整数序列 / Generate integer sequence
pred_pos1 = [0 for _ in range(23)] + [1 for _ in range(77)]
# 生成整数序列 / Generate integer sequence
pred_pos2 = [0 for _ in range(5)] + [2 for _ in range(95)]
# 生成整数序列 / Generate integer sequence
pred_neg = [0 for _ in range(10000)]
y_pred = pred_pos1 + pred_pos2 + pred_neg
# calculate recall
# 计算召回率 = TP / (TP+FN) / Recall = TP / (TP+FN)
recall = recall_score(y_true, y_pred, labels=[1,2], average='micro')
# 打印输出 / Print output
print('Recall: %.3f' % recall)
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Fmeasure



---

### Chapter Summary / 章节总结



---
