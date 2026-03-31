# Python 深度学习 / Deep Learning with Python
## Chapter 15

---

### Json

# 01 — Json / 01 Json

**Chapter 15 — File 1 of 4 / 第15章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **MLP for Pima Indians Dataset Serialize to JSON and HDF5**.

本脚本演示 **MLP for Pima Indians Dataset Serialize to JSON and HDF5**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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
## Step 1 — MLP for Pima Indians Dataset Serialize to JSON and HDF5

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential, model_from_json
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
# 导入操作系统接口 / Import OS interface
import os
```

---
## Step 2 — fix random seed for reproducibility

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
numpy.random.seed(7)
```

---
## Step 3 — load pima indians dataset

```python
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
```

---
## Step 4 — split into input (X) and output (Y) variables

```python
X = dataset[:,0:8]
Y = dataset[:,8]
```

---
## Step 5 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(12, input_shape=(8,), activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
```

---
## Step 6 — Compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 7 — Fit the model

```python
# 训练模型 / Train the model
model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
```

---
## Step 8 — evaluate the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X, Y, verbose=0)
# 打印输出 / Print output
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
```

---
## Step 9 — serialize model to JSON

```python
model_json = model.to_json()
# 打开文件（自动关闭） / Open file (auto-close)
with open("model.json", "w") as json_file:
    json_file.write(model_json)
```

---
## Step 10 — serialize weights to HDF5

```python
model.save_weights("model.h5")
# 打印输出 / Print output
print("Saved model to disk")
```

---
## Step 11 — later...
load json and create model

```python
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
```

---
## Step 12 — load weights into new model

```python
loaded_model.load_weights("model.h5")
# 打印输出 / Print output
print("Loaded model from disk")
```

---
## Step 13 — evaluate loaded model on test data

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                     metrics=['accuracy'])
# 评估模型在测试集上的表现 / Evaluate model on test set
score = loaded_model.evaluate(X, Y, verbose=0)
# 打印输出 / Print output
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
```

---
## Learning Notes / 学习笔记

- **概念**: MLP for Pima Indians Dataset Serialize to JSON and HDF5 是机器学习中的常用技术。  
  *MLP for Pima Indians Dataset Serialize to JSON and HDF5 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Json / 01 Json
# Complete Code / 完整代码
# ===============================

# MLP for Pima Indians Dataset Serialize to JSON and HDF5
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential, model_from_json
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
# 导入操作系统接口 / Import OS interface
import os
# fix random seed for reproducibility
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(12, input_shape=(8,), activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# Compile model
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
# 训练模型 / Train the model
model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
# evaluate the model
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X, Y, verbose=0)
# 打印输出 / Print output
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
# 打开文件（自动关闭） / Open file (auto-close)
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
# 打印输出 / Print output
print("Saved model to disk")

# later...

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
# 打印输出 / Print output
print("Loaded model from disk")

# evaluate loaded model on test data
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                     metrics=['accuracy'])
# 评估模型在测试集上的表现 / Evaluate model on test set
score = loaded_model.evaluate(X, Y, verbose=0)
# 打印输出 / Print output
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Yaml

# 04 — Yaml / 04 Yaml

**Chapter 15 — File 2 of 4 / 第15章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **MLP for Pima Indians Dataset serialize to YAML and HDF5**.

本脚本演示 **MLP for Pima Indians Dataset serialize to YAML and HDF5**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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
## Step 1 — MLP for Pima Indians Dataset serialize to YAML and HDF5

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential, model_from_yaml
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
# 导入操作系统接口 / Import OS interface
import os
```

---
## Step 2 — fix random seed for reproducibility

```python
seed = 7
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
numpy.random.seed(seed)
```

---
## Step 3 — load pima indians dataset

```python
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
```

---
## Step 4 — split into input (X) and output (Y) variables

```python
X = dataset[:,0:8]
Y = dataset[:,8]
```

---
## Step 5 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(12, input_shape=(8,), activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
```

---
## Step 6 — Compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 7 — Fit the model

```python
# 训练模型 / Train the model
model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
```

---
## Step 8 — evaluate the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X, Y, verbose=0)
# 打印输出 / Print output
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
```

---
## Step 9 — serialize model to YAML

```python
model_yaml = model.to_yaml()
# 打开文件（自动关闭） / Open file (auto-close)
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
```

---
## Step 10 — serialize weights to HDF5

```python
model.save_weights("model.h5")
# 打印输出 / Print output
print("Saved model to disk")
```

---
## Step 11 — later...
load YAML and create model

```python
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
```

---
## Step 12 — load weights into new model

```python
loaded_model.load_weights("model.h5")
# 打印输出 / Print output
print("Loaded model from disk")
```

---
## Step 13 — evaluate loaded model on test data

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                     metrics=['accuracy'])
# 评估模型在测试集上的表现 / Evaluate model on test set
score = loaded_model.evaluate(X, Y, verbose=0)
# 打印输出 / Print output
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
```

---
## Learning Notes / 学习笔记

- **概念**: MLP for Pima Indians Dataset serialize to YAML and HDF5 是机器学习中的常用技术。  
  *MLP for Pima Indians Dataset serialize to YAML and HDF5 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Yaml / 04 Yaml
# Complete Code / 完整代码
# ===============================

# MLP for Pima Indians Dataset serialize to YAML and HDF5
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential, model_from_yaml
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
# 导入操作系统接口 / Import OS interface
import os
# fix random seed for reproducibility
seed = 7
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(12, input_shape=(8,), activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# Compile model
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
# 训练模型 / Train the model
model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
# evaluate the model
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X, Y, verbose=0)
# 打印输出 / Print output
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to YAML
model_yaml = model.to_yaml()
# 打开文件（自动关闭） / Open file (auto-close)
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.h5")
# 打印输出 / Print output
print("Saved model to disk")

# later...

# load YAML and create model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")
# 打印输出 / Print output
print("Loaded model from disk")

# evaluate loaded model on test data
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                     metrics=['accuracy'])
# 评估模型在测试集上的表现 / Evaluate model on test set
score = loaded_model.evaluate(X, Y, verbose=0)
# 打印输出 / Print output
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Hdf5

# 06 — Hdf5 / 06 Hdf5

**Chapter 15 — File 3 of 4 / 第15章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **MLP for Pima Indians Dataset saved to single file**.

本脚本演示 **MLP for Pima Indians Dataset saved to single file**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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
## Step 1 — MLP for Pima Indians Dataset saved to single file

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import loadtxt
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
```

---
## Step 2 — load pima indians dataset

```python
dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
```

---
## Step 3 — split into input (X) and output (Y) variables

```python
X = dataset[:,0:8]
Y = dataset[:,8]
```

---
## Step 4 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(12, input_shape=(8,), activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
```

---
## Step 5 — compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 6 — Fit the model

```python
# 训练模型 / Train the model
model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
```

---
## Step 7 — evaluate the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X, Y, verbose=0)
# 打印输出 / Print output
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
```

---
## Step 8 — save model and architecture to single file

```python
# 保存模型到文件 / Save model to file
model.save("model.h5")
# 打印输出 / Print output
print("Saved model to disk")
```

---
## Learning Notes / 学习笔记

- **概念**: MLP for Pima Indians Dataset saved to single file 是机器学习中的常用技术。  
  *MLP for Pima Indians Dataset saved to single file is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Hdf5 / 06 Hdf5
# Complete Code / 完整代码
# ===============================

# MLP for Pima Indians Dataset saved to single file
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import loadtxt
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# load pima indians dataset
dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(12, input_shape=(8,), activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# compile model
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
# 训练模型 / Train the model
model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
# evaluate the model
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X, Y, verbose=0)
# 打印输出 / Print output
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# save model and architecture to single file
# 保存模型到文件 / Save model to file
model.save("model.h5")
# 打印输出 / Print output
print("Saved model to disk")
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Load Model

# 08 — Load Model / 08 Load Model

**Chapter 15 — File 4 of 4 / 第15章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **load and evaluate a saved model**.

本脚本演示 **load and evaluate a saved model**。

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
  🏗️ 定义模型 / Define Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — load and evaluate a saved model

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import loadtxt
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import load_model
```

---
## Step 2 — load model

```python
# 从文件加载模型 / Load model from file
model = load_model('model.h5')
```

---
## Step 3 — summarize model.

```python
model.summary()
```

---
## Step 4 — load dataset

```python
dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
```

---
## Step 5 — split into input (X) and output (Y) variables

```python
X = dataset[:,0:8]
Y = dataset[:,8]
```

---
## Step 6 — evaluate the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
score = model.evaluate(X, Y, verbose=0)
# 打印输出 / Print output
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
```

---
## Learning Notes / 学习笔记

- **概念**: load and evaluate a saved model 是机器学习中的常用技术。  
  *load and evaluate a saved model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Model / 08 Load Model
# Complete Code / 完整代码
# ===============================

# load and evaluate a saved model
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import loadtxt
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import load_model

# load model
# 从文件加载模型 / Load model from file
model = load_model('model.h5')
# summarize model.
model.summary()
# load dataset
dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# evaluate the model
# 评估模型在测试集上的表现 / Evaluate model on test set
score = model.evaluate(X, Y, verbose=0)
# 打印输出 / Print output
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
```

---

### Chapter Summary / 章节总结

# Chapter 15 Summary / 第15章总结

## Theme / 主题: Chapter 15 / Chapter 15

This chapter contains **4 code files** demonstrating chapter 15.

本章包含 **4 个代码文件**，演示Chapter 15。

---
## Evolution / 演化路线

  1. `01_json.ipynb` — Json
  2. `04_yaml.ipynb` — Yaml
  3. `06_hdf5.ipynb` — Hdf5
  4. `08_load_model.ipynb` — Load Model

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 15) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 15）是机器学习流水线中的基础构建块。

---
