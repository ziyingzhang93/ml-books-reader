# 统计方法与机器学习
## Chapter 18

---

### Cross Validation

# 18 — Cross-Validation / 交叉验证

**Chapter 18 — File 1 of 1**

## Summary / 摘要

Cross-validation partitions the dataset into k folds for systematic evaluation of model performance. Each fold is used once as a test set while the remaining k-1 folds serve as training data. Setting shuffle=True randomizes the data before splitting, reducing bias from temporal or sequential patterns. This technique provides a robust estimate of generalization error without requiring a separate validation set.

交叉验证将数据集分成k折进行系统的模型性能评估。每一折被用作一次测试集，而其余的k-1折作为训练数据。设置shuffle=True会在拆分之前随机化数据，减少来自时间或顺序模式的偏差。此技术提供了对泛化误差的稳健估计，无需单独的验证集。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


## Step 1 — Import Libraries / 导入库

```python
# Import required libraries
# 导入所需库
from sklearn.model_selection import KFold
import numpy as np

# Set random seed for reproducibility
# 设置随机种子以保证可重复性
np.random.seed(42)
```

## Step 2 — Generate Sample Dataset / 生成样本数据集

```python
# Generate synthetic dataset
# 生成合成数据集
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
y = np.array([0, 1, 0, 1, 0, 1])

# Display dataset
# 显示数据集
print(f"Dataset size: {len(X)}")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeatures:\n{X}")
print(f"\nTarget: {y}")
```

## Step 3 — Initialize KFold with Shuffle / 初始化带随机化的KFold

```python
# Initialize KFold with 3 splits and shuffle enabled
# 用3次拆分和启用的随机化初始化KFold
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Display KFold configuration
# 显示KFold配置
print(f"KFold configuration:")
print(f"  Number of splits: {kf.n_splits}")
print(f"  Shuffle: {kf.shuffle}")
```

## Step 4 — Enumerate Cross-Validation Splits / 枚举交叉验证拆分

```python
# Enumerate train-test splits
# 枚举训练-测试拆分
fold_count = 1

for train_index, test_index in kf.split(X):
    # Extract training and test sets for this fold
    # 提取此折的训练和测试集
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Print fold information
    # 打印折信息
    print(f"\nFold {fold_count}:")
    print(f"  Train indices: {train_index}")
    print(f"  Test indices: {test_index}")
    print(f"  Train set size: {len(X_train)}")
    print(f"  Test set size: {len(X_test)}")
    
    # Display training data
    # 显示训练数据
    print(f"  X_train:\n{X_train}")
    print(f"  y_train: {y_train}")
    
    # Display test data
    # 显示测试数据
    print(f"  X_test:\n{X_test}")
    print(f"  y_test: {y_test}")
    
    fold_count += 1
```

## Step 5 — Demonstrate with Manual Model Training / 使用手动模型训练演示

```python
# Store fold results
# 存储折结果
fold_accuracies = []
fold_number = 1

# Iterate through cross-validation folds
# 遍历交叉验证折
for train_index, test_index in kf.split(X):
    # Extract training and test data
    # 提取训练和测试数据
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Simple model: predict class 1 if sum > threshold, else 0
    # 简单模型：如果和>阈值，预测类1，否则预测0
    threshold = np.mean(X_train.sum(axis=1))
    
    # Make predictions on test set
    # 在测试集上进行预测
    y_pred = (X_test.sum(axis=1) > threshold).astype(int)
    
    # Calculate accuracy
    # 计算准确性
    accuracy = np.mean(y_pred == y_test)
    fold_accuracies.append(accuracy)
    
    print(f"Fold {fold_number} - Accuracy: {accuracy:.2f}")
    fold_number += 1

# Calculate cross-validation metrics
# 计算交叉验证指标
print(f"\nCross-Validation Results:")
print(f"Mean CV Accuracy: {np.mean(fold_accuracies):.2f}")
print(f"Std CV Accuracy: {np.std(fold_accuracies):.2f}")
```

## Learning Notes / 学习笔记

- **Statistical Concept**: K-fold cross-validation partitions data into k disjoint folds, ensuring each observation appears exactly once in test and k-1 times in training across all folds. The shuffle parameter randomizes fold assignment, reducing variance from data ordering effects.
  
  **统计概念**: k折交叉验证将数据分成k个不相交的折，确保在所有折中，每个观察在测试中出现一次，在训练中出现k-1次。shuffle参数随机化折分配，减少数据排序效应引起的方差。

- **ML Application**: Cross-validation provides robust performance estimates with lower bias than single train-test splits. Essential for hyperparameter tuning (grid search, random search), model selection, and detecting overfitting. Repeated k-fold further reduces variance in noisy datasets.
  
  **ML应用**: 交叉验证提供了比单一训练-测试拆分更低偏差的稳健性能估计。对超参数调整(网格搜索、随机搜索)、模型选择和过度拟合检测至关重要。重复k折进一步减少了嘈杂数据集中的方差。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `np.mean` | 计算均值 | Calculate mean |
| `np.random` | 随机数生成 | Random number generation |
| `np.std` | 计算标准差 | Calculate standard deviation |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

## Complete Code / 完整代码一览

```python
from sklearn.model_selection import KFold
import numpy as np

np.random.seed(42)
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
y = np.array([0, 1, 0, 1, 0, 1])

kf = KFold(n_splits=3, shuffle=True, random_state=42)

fold_accuracies = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    threshold = np.mean(X_train.sum(axis=1))
    y_pred = (X_test.sum(axis=1) > threshold).astype(int)
    
    accuracy = np.mean(y_pred == y_test)
    fold_accuracies.append(accuracy)
    print(f"Fold Accuracy: {accuracy:.2f}")

print(f"Mean CV Accuracy: {np.mean(fold_accuracies):.2f}")
print(f"Std CV Accuracy: {np.std(fold_accuracies):.2f}")
```

---
