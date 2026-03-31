# Python 深度学习 / Deep Learning with Python
## Chapter 09

---

### Iris

# 08 — Iris / 08 Iris

**Chapter 09 — File 1 of 1 / 第09章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **multi-class classification with Keras**.

本脚本演示 **multi-class classification with Keras**。

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
```

---
## Step 1 — multi-class classification with Keras

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
```

---
## Step 2 — load dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = pandas.read_csv("iris.csv", header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# 转换数据类型 / Convert data type
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]
```

---
## Step 3 — encode class values as integers

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(Y)
# 用已拟合的模型转换数据 / Transform data with fitted model
encoded_Y = encoder.transform(Y)
```

---
## Step 4 — convert integers to dummy variables (i.e. one-hot encoded)

```python
dummy_y = to_categorical(encoded_Y)
```

---
## Step 5 — define baseline model

```python
def baseline_model():
```

---
## Step 6 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(8, input_shape=(4,), activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(3, activation='softmax'))
```

---
## Step 7 — Compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(model=baseline_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
# 打印输出 / Print output
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

---
## Learning Notes / 学习笔记

- **概念**: multi-class classification with Keras 是机器学习中的常用技术。  
  *multi-class classification with Keras is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Iris / 08 Iris
# Complete Code / 完整代码
# ===============================

# multi-class classification with Keras
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# load dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = pandas.read_csv("iris.csv", header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# 转换数据类型 / Convert data type
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]
# encode class values as integers
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(Y)
# 用已拟合的模型转换数据 / Transform data with fitted model
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one-hot encoded)
dummy_y = to_categorical(encoded_Y)

# define baseline model
def baseline_model():
    # create model
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(8, input_shape=(4,), activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(3, activation='softmax'))
    # Compile model
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(model=baseline_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
# 打印输出 / Print output
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

---

### Chapter Summary / 章节总结

# Chapter 09 Summary / 第09章总结

## Theme / 主题: Chapter 09 / Chapter 09

This chapter contains **1 code files** demonstrating chapter 09.

本章包含 **1 个代码文件**，演示Chapter 09。

---
## Evolution / 演化路线

  1. `08_iris.ipynb` — Iris

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 09) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 09）是机器学习流水线中的基础构建块。

---
