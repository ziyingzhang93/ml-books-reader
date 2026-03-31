# Python 深度学习 / Deep Learning with Python
## Chapter 11

---

### Baseline

# 06 — Baseline / 06 Baseline

**Chapter 11 — File 1 of 4 / 第11章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Regression Example With Boston Dataset: Baseline**.

本脚本演示 **Regression Example With Boston Dataset: Baseline**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
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
```

---
## Step 1 — Regression Example With Boston Dataset: Baseline

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
```

---
## Step 2 — load dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv("housing.csv", delim_whitespace=True, header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
```

---
## Step 3 — split into input (X) and output (Y) variables

```python
X = dataset[:,0:13]
Y = dataset[:,13]
```

---
## Step 4 — define base model

```python
def baseline_model():
```

---
## Step 5 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(13, input_shape=(13,),
                    kernel_initializer='normal', activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(1, kernel_initializer='normal'))
```

---
## Step 6 — Compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer='adam')
    return model
```

---
## Step 7 — evaluate model

```python
estimator = KerasRegressor(model=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(estimator, X, Y, cv=kfold, scoring='neg_mean_squared_error')
# 打印输出 / Print output
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
```

---
## Learning Notes / 学习笔记

- **概念**: Regression Example With Boston Dataset: Baseline 是机器学习中的常用技术。  
  *Regression Example With Boston Dataset: Baseline is a common technique in machine learning.*

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

# Regression Example With Boston Dataset: Baseline
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# load dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv("housing.csv", delim_whitespace=True, header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]
# define base model
def baseline_model():
    # create model
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(13, input_shape=(13,),
                    kernel_initializer='normal', activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
# evaluate model
estimator = KerasRegressor(model=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(estimator, X, Y, cv=kfold, scoring='neg_mean_squared_error')
# 打印输出 / Print output
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Standardized

# 08 — Standardized / 08 Standardized

**Chapter 11 — File 2 of 4 / 第11章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Regression Example With Boston Dataset: Standardized**.

本脚本演示 **Regression Example With Boston Dataset: Standardized**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
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
```

---
## Step 1 — Regression Example With Boston Dataset: Standardized

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
```

---
## Step 2 — load dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv("housing.csv", delim_whitespace=True, header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
```

---
## Step 3 — split into input (X) and output (Y) variables

```python
X = dataset[:,0:13]
Y = dataset[:,13]
```

---
## Step 4 — define base model

```python
def baseline_model():
```

---
## Step 5 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(13, input_shape=(13,),
                    kernel_initializer='normal', activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(1, kernel_initializer='normal'))
```

---
## Step 6 — Compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer='adam')
    return model
```

---
## Step 7 — evaluate model with standardized dataset

```python
estimators = []
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
estimators.append(('standardize', StandardScaler()))
# 添加元素到列表末尾 / Append element to list end
estimators.append(('mlp', KerasRegressor(model=baseline_model,
                                         epochs=50, batch_size=5, verbose=0)))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(pipeline, X, Y, cv=kfold, scoring='neg_mean_squared_error')
# 打印输出 / Print output
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
```

---
## Learning Notes / 学习笔记

- **概念**: Regression Example With Boston Dataset: Standardized 是机器学习中的常用技术。  
  *Regression Example With Boston Dataset: Standardized is a common technique in machine learning.*

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

# Regression Example With Boston Dataset: Standardized
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# load dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv("housing.csv", delim_whitespace=True, header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]
# define base model
def baseline_model():
    # create model
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(13, input_shape=(13,),
                    kernel_initializer='normal', activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
# evaluate model with standardized dataset
estimators = []
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
estimators.append(('standardize', StandardScaler()))
# 添加元素到列表末尾 / Append element to list end
estimators.append(('mlp', KerasRegressor(model=baseline_model,
                                         epochs=50, batch_size=5, verbose=0)))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(pipeline, X, Y, cv=kfold, scoring='neg_mean_squared_error')
# 打印输出 / Print output
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Larger

# 11 — Larger / 11 Larger

**Chapter 11 — File 3 of 4 / 第11章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Regression Example With Boston Dataset: Standardized and Larger**.

本脚本演示 **Regression Example With Boston Dataset: Standardized and Larger**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
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
```

---
## Step 1 — Regression Example With Boston Dataset: Standardized and Larger

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
```

---
## Step 2 — load dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv("housing.csv", delim_whitespace=True, header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
```

---
## Step 3 — split into input (X) and output (Y) variables

```python
X = dataset[:,0:13]
Y = dataset[:,13]
```

---
## Step 4 — define the model

```python
def larger_model():
```

---
## Step 5 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(13, input_shape=(13,),
                    kernel_initializer='normal', activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(1, kernel_initializer='normal'))
```

---
## Step 6 — Compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer='adam')
    return model
```

---
## Step 7 — evaluate model with standardized dataset

```python
estimators = []
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
estimators.append(('standardize', StandardScaler()))
# 添加元素到列表末尾 / Append element to list end
estimators.append(('mlp', KerasRegressor(model=larger_model,
                                         epochs=50, batch_size=5, verbose=0)))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(pipeline, X, Y, cv=kfold, scoring='neg_mean_squared_error')
# 打印输出 / Print output
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))
```

---
## Learning Notes / 学习笔记

- **概念**: Regression Example With Boston Dataset: Standardized and Larger 是机器学习中的常用技术。  
  *Regression Example With Boston Dataset: Standardized and Larger is a common technique in machine learning.*

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
# Larger / 11 Larger
# Complete Code / 完整代码
# ===============================

# Regression Example With Boston Dataset: Standardized and Larger
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# load dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv("housing.csv", delim_whitespace=True, header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]
# define the model
def larger_model():
    # create model
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(13, input_shape=(13,),
                    kernel_initializer='normal', activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
# evaluate model with standardized dataset
estimators = []
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
estimators.append(('standardize', StandardScaler()))
# 添加元素到列表末尾 / Append element to list end
estimators.append(('mlp', KerasRegressor(model=larger_model,
                                         epochs=50, batch_size=5, verbose=0)))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(pipeline, X, Y, cv=kfold, scoring='neg_mean_squared_error')
# 打印输出 / Print output
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Wider

# 14 — Wider / 14 Wider

**Chapter 11 — File 4 of 4 / 第11章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Regression Example With Boston Dataset: Standardized and Wider**.

本脚本演示 **Regression Example With Boston Dataset: Standardized and Wider**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
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
```

---
## Step 1 — Regression Example With Boston Dataset: Standardized and Wider

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
```

---
## Step 2 — load dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv("housing.csv", delim_whitespace=True, header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
```

---
## Step 3 — split into input (X) and output (Y) variables

```python
X = dataset[:,0:13]
Y = dataset[:,13]
```

---
## Step 4 — define wider model

```python
def wider_model():
```

---
## Step 5 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(20, input_shape=(13,),
                    kernel_initializer='normal', activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(1, kernel_initializer='normal'))
```

---
## Step 6 — Compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer='adam')
    return model
```

---
## Step 7 — evaluate model with standardized dataset

```python
estimators = []
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
estimators.append(('standardize', StandardScaler()))
# 添加元素到列表末尾 / Append element to list end
estimators.append(('mlp', KerasRegressor(model=wider_model,
                                         epochs=100, batch_size=5, verbose=0)))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(pipeline, X, Y, cv=kfold, scoring='neg_mean_squared_error')
# 打印输出 / Print output
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))
```

---
## Learning Notes / 学习笔记

- **概念**: Regression Example With Boston Dataset: Standardized and Wider 是机器学习中的常用技术。  
  *Regression Example With Boston Dataset: Standardized and Wider is a common technique in machine learning.*

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
# Wider / 14 Wider
# Complete Code / 完整代码
# ===============================

# Regression Example With Boston Dataset: Standardized and Wider
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# load dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv("housing.csv", delim_whitespace=True, header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]
# define wider model
def wider_model():
    # create model
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(20, input_shape=(13,),
                    kernel_initializer='normal', activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
# evaluate model with standardized dataset
estimators = []
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
estimators.append(('standardize', StandardScaler()))
# 添加元素到列表末尾 / Append element to list end
estimators.append(('mlp', KerasRegressor(model=wider_model,
                                         epochs=100, batch_size=5, verbose=0)))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(pipeline, X, Y, cv=kfold, scoring='neg_mean_squared_error')
# 打印输出 / Print output
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))
```

---

### Chapter Summary / 章节总结

# Chapter 11 Summary / 第11章总结

## Theme / 主题: Chapter 11 / Chapter 11

This chapter contains **4 code files** demonstrating chapter 11.

本章包含 **4 个代码文件**，演示Chapter 11。

---
## Evolution / 演化路线

  1. `06_baseline.ipynb` — Baseline
  2. `08_standardized.ipynb` — Standardized
  3. `11_larger.ipynb` — Larger
  4. `14_wider.ipynb` — Wider

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 11) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 11）是机器学习流水线中的基础构建块。

---
