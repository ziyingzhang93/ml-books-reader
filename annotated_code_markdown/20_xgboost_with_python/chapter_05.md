# Python XGBoost 实战 / XGBoost with Python
## Chapter 05

---

### Chapter Summary / 章节总结

# Chapter 05 Summary / 第05章总结

## Theme / 主题: Chapter 05 / Chapter 05

This chapter contains **4 code files** demonstrating chapter 05.

本章包含 **4 个代码文件**，演示Chapter 05。

---
## Evolution / 演化路线

  1. `breast_one_hot.ipynb` — Breast One Hot
  2. `horse_colic_missing.ipynb` — Horse Colic Missing
  3. `horse_colic_missing_imputer.ipynb` — Horse Colic Missing Imputer
  4. `iris_label_encode.ipynb` — Iris Label Encode

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 05) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 05）是机器学习流水线中的基础构建块。

---

### Breast One Hot

# 01 — Breast One Hot / Breast One Hot

**Chapter 05 — File 1 of 4 / 第05章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **binary classification, breast cancer dataset, label and one hot encoded**.

本脚本演示 **binary classification, breast cancer dataset, label and one hot encoded**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
  │
  ▼
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
  │
  ▼
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
```

---
## Step 1 — binary classification, breast cancer dataset, label and one hot encoded

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import column_stack
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入XGBoost梯度提升库 / Import XGBoost gradient boosting library
from xgboost import XGBClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OneHotEncoder
```

---
## Step 2 — load data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = read_csv('datasets-uci-breast-cancer.csv', header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = data.values
```

---
## Step 3 — split data into X and y

```python
X = dataset[:,0:9]
# 转换数据类型 / Convert data type
X = X.astype(str)
Y = dataset[:,9]
```

---
## Step 4 — encode string input values as integers

```python
columns = []
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
for i in range(0, X.shape[1]):
 # 将类别标签编码为数字 / Encode categorical labels to numbers
	label_encoder = LabelEncoder()
 # 拟合并转换数据（一步完成） / Fit and transform data (one step)
	feature = label_encoder.fit_transform(X[:,i])
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	feature = feature.reshape(X.shape[0], 1)
 # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
	onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
 # 拟合并转换数据（一步完成） / Fit and transform data (one step)
	feature = onehot_encoder.fit_transform(feature)
 # 添加元素到列表末尾 / Append element to list end
	columns.append(feature)
```

---
## Step 5 — collapse columns into array

```python
encoded_x = column_stack(columns)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("X shape: ", encoded_x.shape)
```

---
## Step 6 — encode string class values as integers

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
# 用已拟合的模型转换数据 / Transform data with fitted model
label_encoded_y = label_encoder.transform(Y)
```

---
## Step 7 — split data into train and test sets

```python
seed = 7
test_size = 0.33
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(encoded_x, label_encoded_y, test_size=test_size, random_state=seed)
```

---
## Step 8 — fit model on training data

```python
model = XGBClassifier()
# 训练模型 / Train the model
model.fit(X_train, y_train)
# 打印输出 / Print output
print(model)
```

---
## Step 9 — make predictions for test data

```python
# 用模型做预测 / Make predictions with model
predictions = model.predict(X_test)
```

---
## Step 10 — evaluate predictions

```python
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
accuracy = accuracy_score(y_test, predictions)
# 打印输出 / Print output
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

---
## Learning Notes / 学习笔记

- **概念**: binary classification, breast cancer dataset, label and one hot encoded 是机器学习中的常用技术。  
  *binary classification, breast cancer dataset, label and one hot encoded is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `XGBClassifier` | XGBoost分类器 | XGBoost classifier |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |
| `xgboost` | 梯度提升框架 | Gradient boosting framework |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Breast One Hot / Breast One Hot
# Complete Code / 完整代码
# ===============================

# binary classification, breast cancer dataset, label and one hot encoded
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import column_stack
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入XGBoost梯度提升库 / Import XGBoost gradient boosting library
from xgboost import XGBClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OneHotEncoder
# load data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = read_csv('datasets-uci-breast-cancer.csv', header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = data.values
# split data into X and y
X = dataset[:,0:9]
# 转换数据类型 / Convert data type
X = X.astype(str)
Y = dataset[:,9]
# encode string input values as integers
columns = []
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
for i in range(0, X.shape[1]):
 # 将类别标签编码为数字 / Encode categorical labels to numbers
	label_encoder = LabelEncoder()
 # 拟合并转换数据（一步完成） / Fit and transform data (one step)
	feature = label_encoder.fit_transform(X[:,i])
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	feature = feature.reshape(X.shape[0], 1)
 # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
	onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
 # 拟合并转换数据（一步完成） / Fit and transform data (one step)
	feature = onehot_encoder.fit_transform(feature)
 # 添加元素到列表末尾 / Append element to list end
	columns.append(feature)
# collapse columns into array
encoded_x = column_stack(columns)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("X shape: ", encoded_x.shape)
# encode string class values as integers
# 将类别标签编码为数字 / Encode categorical labels to numbers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
# 用已拟合的模型转换数据 / Transform data with fitted model
label_encoded_y = label_encoder.transform(Y)
# split data into train and test sets
seed = 7
test_size = 0.33
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(encoded_x, label_encoded_y, test_size=test_size, random_state=seed)
# fit model on training data
model = XGBClassifier()
# 训练模型 / Train the model
model.fit(X_train, y_train)
# 打印输出 / Print output
print(model)
# make predictions for test data
# 用模型做预测 / Make predictions with model
predictions = model.predict(X_test)
# evaluate predictions
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
accuracy = accuracy_score(y_test, predictions)
# 打印输出 / Print output
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Horse Colic Missing



---

### Horse Colic Missing Imputer



---

### Iris Label Encode



---
