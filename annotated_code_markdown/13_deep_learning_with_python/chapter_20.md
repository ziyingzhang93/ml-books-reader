# Python 深度学习 / Deep Learning with Python
## Chapter 20

---

### Baseline



---

### Dropout

# 02 — Dropout / 随机失活

**Chapter 20 — File 2 of 3 / 第20章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Example of Dropout on the Sonar Dataset: Visible Layer**.

本脚本演示 **Example of Dropout on the Sonar Dataset: Visible Layer**。

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
## Step 1 — Example of Dropout on the Sonar Dataset: Visible Layer

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dropout
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.constraints import MaxNorm
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.optimizers import SGD
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
## Step 5 — dropout in the input layer with weight constraint

```python
def create_model():
```

---
## Step 6 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dropout(0.2, input_shape=(60,)))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(60, activation='relu', kernel_constraint=MaxNorm(3)))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(30, activation='relu', kernel_constraint=MaxNorm(3)))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(1, activation='sigmoid'))
```

---
## Step 7 — Compile model

```python
sgd = SGD(learning_rate=0.1, momentum=0.9)
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

estimators = []
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
estimators.append(('standardize', StandardScaler()))
# 添加元素到列表末尾 / Append element to list end
estimators.append(('mlp', KerasClassifier(model=create_model,
                                          epochs=300, batch_size=16, verbose=0)))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
# 打印输出 / Print output
print("Visible: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

---
## Learning Notes / 学习笔记

- **概念**: Example of Dropout on the Sonar Dataset: Visible Layer 是机器学习中的常用技术。  
  *Example of Dropout on the Sonar Dataset: Visible Layer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `learning_rate` | 学习率：参数更新步长 | Learning rate: step size for parameter updates |
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
# Dropout / 随机失活
# Complete Code / 完整代码
# ===============================

# Example of Dropout on the Sonar Dataset: Visible Layer
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dropout
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.constraints import MaxNorm
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.optimizers import SGD
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

# dropout in the input layer with weight constraint
def create_model():
    # create model
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dropout(0.2, input_shape=(60,)))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(60, activation='relu', kernel_constraint=MaxNorm(3)))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(30, activation='relu', kernel_constraint=MaxNorm(3)))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    sgd = SGD(learning_rate=0.1, momentum=0.9)
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

estimators = []
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
estimators.append(('standardize', StandardScaler()))
# 添加元素到列表末尾 / Append element to list end
estimators.append(('mlp', KerasClassifier(model=create_model,
                                          epochs=300, batch_size=16, verbose=0)))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
# 打印输出 / Print output
print("Visible: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Hidden

# 03 — Hidden / 03 Hidden

**Chapter 20 — File 3 of 3 / 第20章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Example of Dropout on the Sonar Dataset: Hidden Layer**.

本脚本演示 **Example of Dropout on the Sonar Dataset: Hidden Layer**。

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
## Step 1 — Example of Dropout on the Sonar Dataset: Hidden Layer

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dropout
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.constraints import MaxNorm
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.optimizers import SGD
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
## Step 5 — dropout in hidden layers with weight constraint

```python
def create_model():
```

---
## Step 6 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(60, input_shape=(60,),
                    activation='relu', kernel_constraint=MaxNorm(3)))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dropout(0.2))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(30, activation='relu', kernel_constraint=MaxNorm(3)))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dropout(0.2))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(1, activation='sigmoid'))
```

---
## Step 7 — Compile model

```python
sgd = SGD(learning_rate=0.1, momentum=0.9)
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

estimators = []
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
estimators.append(('standardize', StandardScaler()))
# 添加元素到列表末尾 / Append element to list end
estimators.append(('mlp', KerasClassifier(model=create_model,
                                          epochs=300, batch_size=16, verbose=0)))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
# 打印输出 / Print output
print("Hidden: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

---
## Learning Notes / 学习笔记

- **概念**: Example of Dropout on the Sonar Dataset: Hidden Layer 是机器学习中的常用技术。  
  *Example of Dropout on the Sonar Dataset: Hidden Layer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `learning_rate` | 学习率：参数更新步长 | Learning rate: step size for parameter updates |
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
# Hidden / 03 Hidden
# Complete Code / 完整代码
# ===============================

# Example of Dropout on the Sonar Dataset: Hidden Layer
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dropout
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.constraints import MaxNorm
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.optimizers import SGD
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

# dropout in hidden layers with weight constraint
def create_model():
    # create model
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(60, input_shape=(60,),
                    activation='relu', kernel_constraint=MaxNorm(3)))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dropout(0.2))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(30, activation='relu', kernel_constraint=MaxNorm(3)))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dropout(0.2))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    sgd = SGD(learning_rate=0.1, momentum=0.9)
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

estimators = []
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
estimators.append(('standardize', StandardScaler()))
# 添加元素到列表末尾 / Append element to list end
estimators.append(('mlp', KerasClassifier(model=create_model,
                                          epochs=300, batch_size=16, verbose=0)))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
# 打印输出 / Print output
print("Hidden: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

---

### Chapter Summary / 章节总结



---
