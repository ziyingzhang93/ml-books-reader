# Python XGBoost 实战 / XGBoost with Python
## Chapter 07

---

### Chapter Summary / 章节总结

# Chapter 07 Summary / 第07章总结

## Theme / 主题: Chapter 07 / Chapter 07

This chapter contains **2 code files** demonstrating chapter 07.

本章包含 **2 个代码文件**，演示Chapter 07。

---
## Evolution / 演化路线

  1. `plot_tree-left-to-right.ipynb` — Plot Tree-Left-To-Right
  2. `plot_tree.ipynb` — Plot Tree

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 07) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 07）是机器学习流水线中的基础构建块。

---

### Plot Tree-Left-To-Right



---

### Plot Tree

# 01 — Plot Tree / 决策树

**Chapter 07 — File 2 of 2 / 第07章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **plot decision tree**.

本脚本演示 **plot decision tree**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — plot decision tree

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import loadtxt
# 导入XGBoost梯度提升库 / Import XGBoost gradient boosting library
from xgboost import XGBClassifier
# 导入XGBoost梯度提升库 / Import XGBoost gradient boosting library
from xgboost import plot_tree
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load data

```python
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
```

---
## Step 3 — split data into X and y

```python
X = dataset[:,0:8]
y = dataset[:,8]
```

---
## Step 4 — fit model on training data

```python
model = XGBClassifier()
# 训练模型 / Train the model
model.fit(X, y)
```

---
## Step 5 — plot single tree

```python
plot_tree(model)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: plot decision tree 是机器学习中的常用技术。  
  *plot decision tree is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `XGBClassifier` | XGBoost分类器 | XGBoost classifier |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `xgboost` | 梯度提升框架 | Gradient boosting framework |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot Tree / 决策树
# Complete Code / 完整代码
# ===============================

# plot decision tree
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import loadtxt
# 导入XGBoost梯度提升库 / Import XGBoost gradient boosting library
from xgboost import XGBClassifier
# 导入XGBoost梯度提升库 / Import XGBoost gradient boosting library
from xgboost import plot_tree
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
y = dataset[:,8]
# fit model on training data
model = XGBClassifier()
# 训练模型 / Train the model
model.fit(X, y)
# plot single tree
plot_tree(model)
pyplot.show()
```

---
