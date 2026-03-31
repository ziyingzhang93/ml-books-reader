# Python 深度学习 / Deep Learning with Python
## Chapter 10

---

### Baseline

# 06 — Baseline / 06 Baseline

**Chapter 10 — File 1 of 4 / 第10章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Binary Classification with Sonar Dataset: Baseline**.

本脚本演示 **Binary Classification with Sonar Dataset: Baseline**。

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
## Step 1 — Binary Classification with Sonar Dataset: Baseline

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import StratifiedKFold
```

---
## Step 2 — load dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv("sonar.csv", header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
```

---
## Step 3 — split into input (X) and output (Y) variables

```python
# 转换数据类型 / Convert data type
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
```

---
## Step 4 — encode class values as integers

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(Y)
# 用已拟合的模型转换数据 / Transform data with fitted model
encoded_Y = encoder.transform(Y)
```

---
## Step 5 — baseline model

```python
def create_baseline():
```

---
## Step 6 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(60, input_shape=(60,), activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(1, activation='sigmoid'))
```

---
## Step 7 — Compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
```

---
## Step 8 — evaluate model with standardized dataset

```python
estimator = KerasClassifier(model=create_baseline, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# 打印输出 / Print output
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

---
## Learning Notes / 学习笔记

- **概念**: Binary Classification with Sonar Dataset: Baseline 是机器学习中的常用技术。  
  *Binary Classification with Sonar Dataset: Baseline is a common technique in machine learning.*

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
# Baseline / 06 Baseline
# Complete Code / 完整代码
# ===============================

# Binary Classification with Sonar Dataset: Baseline
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import StratifiedKFold
# load dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv("sonar.csv", header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# split into input (X) and output (Y) variables
# 转换数据类型 / Convert data type
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
# encode class values as integers
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(Y)
# 用已拟合的模型转换数据 / Transform data with fitted model
encoded_Y = encoder.transform(Y)
# baseline model
def create_baseline():
    # create model
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(60, input_shape=(60,), activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# evaluate model with standardized dataset
estimator = KerasClassifier(model=create_baseline, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# 打印输出 / Print output
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Standardized

# 08 — Standardized / 08 Standardized

**Chapter 10 — File 2 of 4 / 第10章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Binary Classification with Sonar Dataset: Standardized**.

本脚本演示 **Binary Classification with Sonar Dataset: Standardized**。

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
## Step 1 — Binary Classification with Sonar Dataset: Standardized

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import StratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
```

---
## Step 2 — load dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv("sonar.csv", header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
```

---
## Step 3 — split into input (X) and output (Y) variables

```python
# 转换数据类型 / Convert data type
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
```

---
## Step 4 — encode class values as integers

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(Y)
# 用已拟合的模型转换数据 / Transform data with fitted model
encoded_Y = encoder.transform(Y)
```

---
## Step 5 — baseline model

```python
def create_baseline():
```

---
## Step 6 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(60, input_shape=(60,), activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(1, activation='sigmoid'))
```

---
## Step 7 — Compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
```

---
## Step 8 — evaluate baseline model with standardized dataset

```python
estimators = []
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
estimators.append(('standardize', StandardScaler()))
# 添加元素到列表末尾 / Append element to list end
estimators.append(('mlp', KerasClassifier(model=create_baseline,
                                          epochs=100, batch_size=5, verbose=0)))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
# 打印输出 / Print output
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

---
## Learning Notes / 学习笔记

- **概念**: Binary Classification with Sonar Dataset: Standardized 是机器学习中的常用技术。  
  *Binary Classification with Sonar Dataset: Standardized is a common technique in machine learning.*

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
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
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
# Standardized / 08 Standardized
# Complete Code / 完整代码
# ===============================

# Binary Classification with Sonar Dataset: Standardized
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import StratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# load dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv("sonar.csv", header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# split into input (X) and output (Y) variables
# 转换数据类型 / Convert data type
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
# encode class values as integers
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(Y)
# 用已拟合的模型转换数据 / Transform data with fitted model
encoded_Y = encoder.transform(Y)
# baseline model
def create_baseline():
    # create model
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(60, input_shape=(60,), activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# evaluate baseline model with standardized dataset
estimators = []
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
estimators.append(('standardize', StandardScaler()))
# 添加元素到列表末尾 / Append element to list end
estimators.append(('mlp', KerasClassifier(model=create_baseline,
                                          epochs=100, batch_size=5, verbose=0)))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
# 打印输出 / Print output
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Smaller Model

# 10 — Smaller Model / 10 Smaller Model

**Chapter 10 — File 3 of 4 / 第10章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Binary Classification with Sonar Dataset: Standardized Smaller**.

本脚本演示 **Binary Classification with Sonar Dataset: Standardized Smaller**。

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
## Step 1 — Binary Classification with Sonar Dataset: Standardized Smaller

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import StratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
```

---
## Step 2 — load dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv("sonar.csv", header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
```

---
## Step 3 — split into input (X) and output (Y) variables

```python
# 转换数据类型 / Convert data type
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
```

---
## Step 4 — encode class values as integers

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(Y)
# 用已拟合的模型转换数据 / Transform data with fitted model
encoded_Y = encoder.transform(Y)
```

---
## Step 5 — smaller model

```python
def create_smaller():
```

---
## Step 6 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(30, input_shape=(60,), activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(1, activation='sigmoid'))
```

---
## Step 7 — Compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
estimators = []
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
estimators.append(('standardize', StandardScaler()))
# 添加元素到列表末尾 / Append element to list end
estimators.append(('mlp', KerasClassifier(model=create_smaller,
                                          epochs=100, batch_size=5, verbose=0)))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
# 打印输出 / Print output
print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

---
## Learning Notes / 学习笔记

- **概念**: Binary Classification with Sonar Dataset: Standardized Smaller 是机器学习中的常用技术。  
  *Binary Classification with Sonar Dataset: Standardized Smaller is a common technique in machine learning.*

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
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
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
# Smaller Model / 10 Smaller Model
# Complete Code / 完整代码
# ===============================

# Binary Classification with Sonar Dataset: Standardized Smaller
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import StratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# load dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv("sonar.csv", header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# split into input (X) and output (Y) variables
# 转换数据类型 / Convert data type
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
# encode class values as integers
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(Y)
# 用已拟合的模型转换数据 / Transform data with fitted model
encoded_Y = encoder.transform(Y)
# smaller model
def create_smaller():
    # create model
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(30, input_shape=(60,), activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
estimators = []
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
estimators.append(('standardize', StandardScaler()))
# 添加元素到列表末尾 / Append element to list end
estimators.append(('mlp', KerasClassifier(model=create_smaller,
                                          epochs=100, batch_size=5, verbose=0)))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
# 打印输出 / Print output
print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Larger Model

# 12 — Larger Model / 12 Larger Model

**Chapter 10 — File 4 of 4 / 第10章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Binary Classification with Sonar Dataset: Standardized Larger**.

本脚本演示 **Binary Classification with Sonar Dataset: Standardized Larger**。

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
## Step 1 — Binary Classification with Sonar Dataset: Standardized Larger

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import StratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
```

---
## Step 2 — load dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv("sonar.csv", header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
```

---
## Step 3 — split into input (X) and output (Y) variables

```python
# 转换数据类型 / Convert data type
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
```

---
## Step 4 — encode class values as integers

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(Y)
# 用已拟合的模型转换数据 / Transform data with fitted model
encoded_Y = encoder.transform(Y)
```

---
## Step 5 — larger model

```python
def create_larger():
```

---
## Step 6 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(60, input_shape=(60,), activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(30, activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(1, activation='sigmoid'))
```

---
## Step 7 — Compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
estimators = []
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
estimators.append(('standardize', StandardScaler()))
# 添加元素到列表末尾 / Append element to list end
estimators.append(('mlp', KerasClassifier(model=create_larger,
                                          epochs=100, batch_size=5, verbose=0)))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
# 打印输出 / Print output
print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

---
## Learning Notes / 学习笔记

- **概念**: Binary Classification with Sonar Dataset: Standardized Larger 是机器学习中的常用技术。  
  *Binary Classification with Sonar Dataset: Standardized Larger is a common technique in machine learning.*

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
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
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
# Larger Model / 12 Larger Model
# Complete Code / 完整代码
# ===============================

# Binary Classification with Sonar Dataset: Standardized Larger
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import StratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# load dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv("sonar.csv", header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# split into input (X) and output (Y) variables
# 转换数据类型 / Convert data type
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
# encode class values as integers
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(Y)
# 用已拟合的模型转换数据 / Transform data with fitted model
encoded_Y = encoder.transform(Y)
# larger model
def create_larger():
    # create model
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(60, input_shape=(60,), activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(30, activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
estimators = []
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
estimators.append(('standardize', StandardScaler()))
# 添加元素到列表末尾 / Append element to list end
estimators.append(('mlp', KerasClassifier(model=create_larger,
                                          epochs=100, batch_size=5, verbose=0)))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
# 打印输出 / Print output
print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

---

### Chapter Summary / 章节总结

# Chapter 10 Summary / 第10章总结

## Theme / 主题: Chapter 10 / Chapter 10

This chapter contains **4 code files** demonstrating chapter 10.

本章包含 **4 个代码文件**，演示Chapter 10。

---
## Evolution / 演化路线

  1. `06_baseline.ipynb` — Baseline
  2. `08_standardized.ipynb` — Standardized
  3. `10_smaller_model.ipynb` — Smaller Model
  4. `12_larger_model.ipynb` — Larger Model

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 10) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 10）是机器学习流水线中的基础构建块。

---
