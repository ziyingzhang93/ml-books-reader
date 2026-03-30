# CV深度学习
## Chapter 08

---

### Load Dataset

# 01 — Load Dataset / 01 Load Dataset

**Chapter 08 — File 1 of 1 / 第08章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **example of progressively loading images from file**.

本脚本演示 **example of progressively loading images from file**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of progressively loading images from file

```python
from keras.preprocessing.image import ImageDataGenerator
```

---
## Step 2 — create generator

```python
datagen = ImageDataGenerator()
```

---
## Step 3 — prepare an iterators for each dataset

```python
train_it = datagen.flow_from_directory('data/train/', class_mode='binary')
val_it = datagen.flow_from_directory('data/validation/', class_mode='binary')
test_it = datagen.flow_from_directory('data/test/', class_mode='binary')
```

---
## Step 4 — confirm the iterator works

```python
batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
```

---
## Learning Notes / 学习笔记

- **概念**: example of progressively loading images from file 是机器学习中的常用技术。  
  *example of progressively loading images from file is a common technique in machine learning.*

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
# Load Dataset / 01 Load Dataset
# Complete Code / 完整代码
# ===============================

# example of progressively loading images from file
from keras.preprocessing.image import ImageDataGenerator
# create generator
datagen = ImageDataGenerator()
# prepare an iterators for each dataset
train_it = datagen.flow_from_directory('data/train/', class_mode='binary')
val_it = datagen.flow_from_directory('data/validation/', class_mode='binary')
test_it = datagen.flow_from_directory('data/test/', class_mode='binary')
# confirm the iterator works
batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
```

---

### Chapter Summary

# Chapter 08 Summary / 第08章总结

## Theme / 主题: Chapter 08 / Chapter 08

This chapter contains **1 code files** demonstrating chapter 08.

本章包含 **1 个代码文件**，演示Chapter 08。

---
## Evolution / 演化路线

  1. `01_load_dataset.ipynb` — Load Dataset

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 08) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 08）是机器学习流水线中的基础构建块。

---
