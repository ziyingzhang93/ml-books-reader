# Python 深度学习 / Deep Learning with Python
## Chapter 16

---

### Checkpoint

# 01 — Checkpoint / 01 Checkpoint

**Chapter 16 — File 1 of 4 / 第16章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Checkpoint the weights when validation accuracy improves**.

本脚本演示 **Checkpoint the weights when validation accuracy improves**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
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
## Step 1 — Checkpoint the weights when validation accuracy improves

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import ModelCheckpoint
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(42)
```

---
## Step 2 — load pima indians dataset

```python
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
```

---
## Step 3 — split into input (X) and output (Y) variables

```python
X = dataset[:,0:8]
Y = dataset[:,8]
```

---
## Step 4 — create model

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
## Step 5 — Compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 6 — checkpoint

```python
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
# 模型检查点：训练中保存最佳模型 / ModelCheckpoint: save best model during training
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')
```

---
## Step 7 — Fit the model

```python
# 训练模型 / Train the model
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10,
          callbacks=[checkpoint], verbose=0)
```

---
## Learning Notes / 学习笔记

- **概念**: Checkpoint the weights when validation accuracy improves 是机器学习中的常用技术。  
  *Checkpoint the weights when validation accuracy improves is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `ModelCheckpoint` | 保存最佳模型 | Save best model during training |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Checkpoint / 01 Checkpoint
# Complete Code / 完整代码
# ===============================

# Checkpoint the weights when validation accuracy improves
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import ModelCheckpoint
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(42)
# load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
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
# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
# 模型检查点：训练中保存最佳模型 / ModelCheckpoint: save best model during training
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')
# Fit the model
# 训练模型 / Train the model
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10,
          callbacks=[checkpoint], verbose=0)
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Best Only

# 02 — Best Only / 02 Best Only

**Chapter 16 — File 2 of 4 / 第16章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Checkpoint the weights for best model on validation accuracy**.

本脚本演示 **Checkpoint the weights for best model on validation accuracy**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
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
## Step 1 — Checkpoint the weights for best model on validation accuracy

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import ModelCheckpoint
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
```

---
## Step 2 — load pima indians dataset

```python
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
```

---
## Step 3 — split into input (X) and output (Y) variables

```python
X = dataset[:,0:8]
Y = dataset[:,8]
```

---
## Step 4 — create model

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
## Step 5 — Compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 6 — checkpoint

```python
filepath="weights.best.hdf5"
# 模型检查点：训练中保存最佳模型 / ModelCheckpoint: save best model during training
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')
callbacks_list = [checkpoint]
```

---
## Step 7 — Fit the model

```python
# 训练模型 / Train the model
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10,
          callbacks=callbacks_list, verbose=0)
```

---
## Learning Notes / 学习笔记

- **概念**: Checkpoint the weights for best model on validation accuracy 是机器学习中的常用技术。  
  *Checkpoint the weights for best model on validation accuracy is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `ModelCheckpoint` | 保存最佳模型 | Save best model during training |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Best Only / 02 Best Only
# Complete Code / 完整代码
# ===============================

# Checkpoint the weights for best model on validation accuracy
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import ModelCheckpoint
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
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
# checkpoint
filepath="weights.best.hdf5"
# 模型检查点：训练中保存最佳模型 / ModelCheckpoint: save best model during training
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# Fit the model
# 训练模型 / Train the model
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10,
          callbacks=callbacks_list, verbose=0)
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Earlystop

# 03 — Earlystop / 03 Earlystop

**Chapter 16 — File 3 of 4 / 第16章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Checkpoint the weights for best model on validation accuracy**.

本脚本演示 **Checkpoint the weights for best model on validation accuracy**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
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
## Step 1 — Checkpoint the weights for best model on validation accuracy

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
```

---
## Step 2 — load pima indians dataset

```python
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
```

---
## Step 3 — split into input (X) and output (Y) variables

```python
X = dataset[:,0:8]
Y = dataset[:,8]
```

---
## Step 4 — create model

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
## Step 5 — Compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 6 — checkpoint

```python
filepath="weights.best.hdf5"
# 模型检查点：训练中保存最佳模型 / ModelCheckpoint: save best model during training
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')
# 早停：验证集不再提升时自动停止训练 / EarlyStopping: stop when validation stops improving
es = EarlyStopping(monitor='val_accuracy', patience=5)
callbacks_list = [checkpoint, es]
```

---
## Step 7 — Fit the model

```python
# 训练模型 / Train the model
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10,
          callbacks=callbacks_list, verbose=0)
```

---
## Learning Notes / 学习笔记

- **概念**: Checkpoint the weights for best model on validation accuracy 是机器学习中的常用技术。  
  *Checkpoint the weights for best model on validation accuracy is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `EarlyStopping` | 早停：验证集不再提升时停止训练 | Stop when validation stops improving |
| `ModelCheckpoint` | 保存最佳模型 | Save best model during training |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Earlystop / 03 Earlystop
# Complete Code / 完整代码
# ===============================

# Checkpoint the weights for best model on validation accuracy
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
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
# checkpoint
filepath="weights.best.hdf5"
# 模型检查点：训练中保存最佳模型 / ModelCheckpoint: save best model during training
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')
# 早停：验证集不再提升时自动停止训练 / EarlyStopping: stop when validation stops improving
es = EarlyStopping(monitor='val_accuracy', patience=5)
callbacks_list = [checkpoint, es]
# Fit the model
# 训练模型 / Train the model
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10,
          callbacks=callbacks_list, verbose=0)
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Load Model

# 04 — Load Model / 04 Load Model

**Chapter 16 — File 4 of 4 / 第16章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **How to load and use weights from a checkpoint**.

本脚本演示 **How to load and use weights from a checkpoint**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — How to load and use weights from a checkpoint

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import ModelCheckpoint
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
```

---
## Step 2 — create model

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
## Step 3 — load weights

```python
model.load_weights("weights.best.hdf5")
```

---
## Step 4 — Compile model (required to make predictions)

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 打印输出 / Print output
print("Created model and loaded weights from file")
```

---
## Step 5 — load pima indians dataset

```python
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
```

---
## Step 6 — split into input (X) and output (Y) variables

```python
X = dataset[:,0:8]
Y = dataset[:,8]
```

---
## Step 7 — estimate accuracy on whole dataset using loaded weights

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X, Y, verbose=0)
# 打印输出 / Print output
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
```

---
## Learning Notes / 学习笔记

- **概念**: How to load and use weights from a checkpoint 是机器学习中的常用技术。  
  *How to load and use weights from a checkpoint is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `ModelCheckpoint` | 保存最佳模型 | Save best model during training |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Model / 04 Load Model
# Complete Code / 完整代码
# ===============================

# How to load and use weights from a checkpoint
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import ModelCheckpoint
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# create model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(12, input_shape=(8,), activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# load weights
model.load_weights("weights.best.hdf5")
# Compile model (required to make predictions)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 打印输出 / Print output
print("Created model and loaded weights from file")
# load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# estimate accuracy on whole dataset using loaded weights
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X, Y, verbose=0)
# 打印输出 / Print output
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
```

---

### Chapter Summary / 章节总结

# Chapter 16 Summary / 第16章总结

## Theme / 主题: Chapter 16 / Chapter 16

This chapter contains **4 code files** demonstrating chapter 16.

本章包含 **4 个代码文件**，演示Chapter 16。

---
## Evolution / 演化路线

  1. `01_checkpoint.ipynb` — Checkpoint
  2. `02_best_only.ipynb` — Best Only
  3. `03_earlystop.ipynb` — Earlystop
  4. `04_load_model.ipynb` — Load Model

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 16) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 16）是机器学习流水线中的基础构建块。

---
