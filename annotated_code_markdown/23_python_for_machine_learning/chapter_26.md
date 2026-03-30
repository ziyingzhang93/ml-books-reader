# Python ML
## Chapter 26

---

### Imshow

# 03 — Imshow / 03 Imshow

**Chapter 26 — File 1 of 11 / 第26章 — 第1个文件（共11个）**

---

## Summary / 总结

This script demonstrates **load dataset**.

本脚本演示 **load dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Step 1

```python
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
```

---
## Step 2 — load dataset

```python
(x_train, train_labels), (_, _) = mnist.load_data()
```

---
## Step 3 — Choose only the digits 0, 1, 2

```python
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
```

---
## Step 4 — Shape of training data

```python
total_examples, img_length, img_width = x_train.shape
```

---
## Step 5 — Print the statistics

```python
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)
```

---
## Step 6 — Show images

```python
img_per_row = 8
fig,ax = plt.subplots(nrows=2, ncols=img_per_row,
                      figsize=(18,4),
                      subplot_kw=dict(xticks=[], yticks=[]))
for row in [0, 1]:
    for col in range(img_per_row):
        ax[row, col].imshow(x_train[row*img_per_row + col].astype('int'))
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: load dataset 是机器学习中的常用技术。  
  *load dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Imshow / 03 Imshow
# Complete Code / 完整代码
# ===============================

from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# load dataset
(x_train, train_labels), (_, _) = mnist.load_data()
# Choose only the digits 0, 1, 2
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
# Shape of training data
total_examples, img_length, img_width = x_train.shape
# Print the statistics
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)

# Show images
img_per_row = 8
fig,ax = plt.subplots(nrows=2, ncols=img_per_row,
                      figsize=(18,4),
                      subplot_kw=dict(xticks=[], yticks=[]))
for row in [0, 1]:
    for col in range(img_per_row):
        ax[row, col].imshow(x_train[row*img_per_row + col].astype('int'))
plt.show()
```

---

➡️ **Next / 下一步**: File 2 of 11

---

### Eigen

# 04 — Eigen / 04 Eigen

**Chapter 26 — File 2 of 11 / 第26章 — 第2个文件（共11个）**

---

## Summary / 总结

This script demonstrates **load dataset**.

本脚本演示 **load dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
from tensorflow.keras.datasets import mnist
from tensorflow import dtypes, tensordot
from tensorflow import convert_to_tensor, linalg, transpose
import numpy as np
```

---
## Step 2 — load dataset

```python
(x_train, train_labels), (_, _) = mnist.load_data()
```

---
## Step 3 — Choose only the digits 0, 1, 2

```python
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
```

---
## Step 4 — Convert the dataset into a 2D array of shape 18623 x 784

```python
x = convert_to_tensor(np.reshape(x_train, (x_train.shape[0], -1)),
                      dtype=dtypes.float32)
```

---
## Step 5 — Eigen-decomposition from a 784 x 784 matrix

```python
eigenvalues, eigenvectors = linalg.eigh(tensordot(transpose(x), x, axes=1))
```

---
## Step 6 — Print the three largest eigenvalues

```python
print('3 largest eigenvalues: ', eigenvalues[-3:])
```

---
## Step 7 — Project the data to eigenvectors

```python
x_pca = tensordot(x, eigenvectors, axes=1)
```

---
## Learning Notes / 学习笔记

- **概念**: load dataset 是机器学习中的常用技术。  
  *load dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `PCA` | 主成分分析，降维 | Principal Component Analysis, dimensionality reduction |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Eigen / 04 Eigen
# Complete Code / 完整代码
# ===============================

from tensorflow.keras.datasets import mnist
from tensorflow import dtypes, tensordot
from tensorflow import convert_to_tensor, linalg, transpose
import numpy as np

# load dataset
(x_train, train_labels), (_, _) = mnist.load_data()
# Choose only the digits 0, 1, 2
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]

# Convert the dataset into a 2D array of shape 18623 x 784
x = convert_to_tensor(np.reshape(x_train, (x_train.shape[0], -1)),
                      dtype=dtypes.float32)
# Eigen-decomposition from a 784 x 784 matrix
eigenvalues, eigenvectors = linalg.eigh(tensordot(transpose(x), x, axes=1))
# Print the three largest eigenvalues
print('3 largest eigenvalues: ', eigenvalues[-3:])
# Project the data to eigenvectors
x_pca = tensordot(x, eigenvectors, axes=1)
```

---

➡️ **Next / 下一步**: File 3 of 11

---

### Scatter

# 06 — Scatter / 06 Scatter

**Chapter 26 — File 3 of 11 / 第26章 — 第3个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Load dataset**.

本脚本演示 **Load dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Step 1

```python
from tensorflow.keras.datasets import mnist
from tensorflow import dtypes, tensordot
from tensorflow import convert_to_tensor, linalg, transpose
import numpy as np
import matplotlib.pyplot as plt
```

---
## Step 2 — Load dataset

```python
(x_train, train_labels), (_, _) = mnist.load_data()
```

---
## Step 3 — Choose only the digits 0, 1, 2

```python
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
```

---
## Step 4 — Verify the shape of training data

```python
total_examples, img_length, img_width = x_train.shape
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)
```

---
## Step 5 — Convert the dataset into a 2D array of shape 18623 x 784

```python
x = convert_to_tensor(np.reshape(x_train, (x_train.shape[0], -1)),
                      dtype=dtypes.float32)
```

---
## Step 6 — Eigen-decomposition from a 784 x 784 matrix

```python
eigenvalues, eigenvectors = linalg.eigh(tensordot(transpose(x), x, axes=1))
```

---
## Step 7 — Print the three largest eigenvalues

```python
print('3 largest eigenvalues: ', eigenvalues[-3:])
```

---
## Step 8 — Project the data to eigenvectors

```python
x_pca = tensordot(x, eigenvectors, axes=1)
```

---
## Step 9 — Create the plot

```python
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(x_pca[:, -1], x_pca[:, -2], c=train_labels, s=5)
legend_plt = ax.legend(*scatter.legend_elements(),
                       loc="lower left", title="Digits")
ax.add_artist(legend_plt)
plt.title('First Two Dimensions of Projected Data After Applying PCA')
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Load dataset 是机器学习中的常用技术。  
  *Load dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `PCA` | 主成分分析，降维 | Principal Component Analysis, dimensionality reduction |
| `matplotlib` | 绑图库 | Plotting library |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Scatter / 06 Scatter
# Complete Code / 完整代码
# ===============================

from tensorflow.keras.datasets import mnist
from tensorflow import dtypes, tensordot
from tensorflow import convert_to_tensor, linalg, transpose
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
(x_train, train_labels), (_, _) = mnist.load_data()
# Choose only the digits 0, 1, 2
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
# Verify the shape of training data
total_examples, img_length, img_width = x_train.shape
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)

# Convert the dataset into a 2D array of shape 18623 x 784
x = convert_to_tensor(np.reshape(x_train, (x_train.shape[0], -1)),
                      dtype=dtypes.float32)
# Eigen-decomposition from a 784 x 784 matrix
eigenvalues, eigenvectors = linalg.eigh(tensordot(transpose(x), x, axes=1))
# Print the three largest eigenvalues
print('3 largest eigenvalues: ', eigenvalues[-3:])
# Project the data to eigenvectors
x_pca = tensordot(x, eigenvectors, axes=1)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(x_pca[:, -1], x_pca[:, -2], c=train_labels, s=5)
legend_plt = ax.legend(*scatter.legend_elements(),
                       loc="lower left", title="Digits")
ax.add_artist(legend_plt)
plt.title('First Two Dimensions of Projected Data After Applying PCA')
plt.show()
```

---

➡️ **Next / 下一步**: File 4 of 11

---

### 3D Scatter

# 09 — 3D Scatter / 09 3D Scatter

**Chapter 26 — File 4 of 11 / 第26章 — 第4个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Load dataset**.

本脚本演示 **Load dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Step 1

```python
from tensorflow.keras.datasets import mnist
from tensorflow import dtypes, tensordot
from tensorflow import convert_to_tensor, linalg, transpose
import numpy as np
import matplotlib.pyplot as plt
```

---
## Step 2 — Load dataset

```python
(x_train, train_labels), (_, _) = mnist.load_data()
```

---
## Step 3 — Choose only the digits 0, 1, 2

```python
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
```

---
## Step 4 — Verify the shape of training data

```python
total_examples, img_length, img_width = x_train.shape
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)
```

---
## Step 5 — Convert the dataset into a 2D array of shape 18623 x 784

```python
x = convert_to_tensor(np.reshape(x_train, (x_train.shape[0], -1)),
                      dtype=dtypes.float32)
```

---
## Step 6 — Eigen-decomposition from a 784 x 784 matrix

```python
eigenvalues, eigenvectors = linalg.eigh(tensordot(transpose(x), x, axes=1))
```

---
## Step 7 — Print the three largest eigenvalues

```python
print('3 largest eigenvalues: ', eigenvalues[-3:])
```

---
## Step 8 — Project the data to eigenvectors

```python
x_pca = tensordot(x, eigenvectors, axes=1)
```

---
## Step 9 — Create the plot

```python
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection='3d')
ax.view_init(elev=30, azim=-60)
plt_3d = ax.scatter3D(x_pca[:, -1], x_pca[:, -2], x_pca[:, -3], c=train_labels, s=1)
plt.colorbar(plt_3d)
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Load dataset 是机器学习中的常用技术。  
  *Load dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `PCA` | 主成分分析，降维 | Principal Component Analysis, dimensionality reduction |
| `matplotlib` | 绑图库 | Plotting library |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# 3D Scatter / 09 3D Scatter
# Complete Code / 完整代码
# ===============================

from tensorflow.keras.datasets import mnist
from tensorflow import dtypes, tensordot
from tensorflow import convert_to_tensor, linalg, transpose
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
(x_train, train_labels), (_, _) = mnist.load_data()
# Choose only the digits 0, 1, 2
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
# Verify the shape of training data
total_examples, img_length, img_width = x_train.shape
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)

# Convert the dataset into a 2D array of shape 18623 x 784
x = convert_to_tensor(np.reshape(x_train, (x_train.shape[0], -1)),
                      dtype=dtypes.float32)
# Eigen-decomposition from a 784 x 784 matrix
eigenvalues, eigenvectors = linalg.eigh(tensordot(transpose(x), x, axes=1))
# Print the three largest eigenvalues
print('3 largest eigenvalues: ', eigenvalues[-3:])
# Project the data to eigenvectors
x_pca = tensordot(x, eigenvectors, axes=1)

# Create the plot
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection='3d')
ax.view_init(elev=30, azim=-60)
plt_3d = ax.scatter3D(x_pca[:, -1], x_pca[:, -2], x_pca[:, -3], c=train_labels, s=1)
plt.colorbar(plt_3d)
plt.show()
```

---

➡️ **Next / 下一步**: File 5 of 11

---

### Seaborn Lines

# 23 — Seaborn Lines / 23 Seaborn Lines

**Chapter 26 — File 8 of 11 / 第26章 — 第8个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Load dataset**.

本脚本演示 **Load dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
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
## Step 1 — Step 1

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

---
## Step 2 — Load dataset

```python
(x_train, train_labels), (_, _) = mnist.load_data()
```

---
## Step 3 — Choose only the digits 0, 1, 2

```python
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
```

---
## Step 4 — Verify the shape of training data

```python
total_examples, img_length, img_width = x_train.shape
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)
```

---
## Step 5 — Prepare for classifier network

```python
epochs = 10
y_train = utils.to_categorical(train_labels)
input_dim = img_length*img_width
```

---
## Step 6 — Create a Sequential model

```python
model = Sequential()
```

---
## Step 7 — First layer for reshaping input images from 2D to 1D

```python
model.add(Reshape((input_dim, ), input_shape=(img_length, img_width)))
```

---
## Step 8 — Dense layer of 8 neurons

```python
model.add(Dense(8, activation='relu'))
```

---
## Step 9 — Output layer

```python
model.add(Dense(total_classes, activation='softmax'))
```

---
## Step 10 — Compile model

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.33,
                    epochs=epochs, batch_size=10, verbose=0)
```

---
## Step 11 — Prepare pandas DataFrame

```python
df_history = pd.DataFrame(history.history)
print(df_history)
```

---
## Step 12 — Plot loss in seaborn

```python
my_plot = sns.lineplot(data=df_history[["loss","val_loss"]])
my_plot.set_xlabel('Epochs')
my_plot.set_ylabel('Loss')
plt.legend(labels=["Training", "Validation"])
plt.title('Training and Validation Loss')
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Load dataset 是机器学习中的常用技术。  
  *Load dataset is a common technique in machine learning.*

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
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Seaborn Lines / 23 Seaborn Lines
# Complete Code / 完整代码
# ===============================

from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
(x_train, train_labels), (_, _) = mnist.load_data()
# Choose only the digits 0, 1, 2
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
# Verify the shape of training data
total_examples, img_length, img_width = x_train.shape
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)

# Prepare for classifier network
epochs = 10
y_train = utils.to_categorical(train_labels)
input_dim = img_length*img_width
# Create a Sequential model
model = Sequential()
# First layer for reshaping input images from 2D to 1D
model.add(Reshape((input_dim, ), input_shape=(img_length, img_width)))
# Dense layer of 8 neurons
model.add(Dense(8, activation='relu'))
# Output layer
model.add(Dense(total_classes, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.33,
                    epochs=epochs, batch_size=10, verbose=0)

# Prepare pandas DataFrame
df_history = pd.DataFrame(history.history)
print(df_history)

# Plot loss in seaborn
my_plot = sns.lineplot(data=df_history[["loss","val_loss"]])
my_plot.set_xlabel('Epochs')
my_plot.set_ylabel('Loss')
plt.legend(labels=["Training", "Validation"])
plt.title('Training and Validation Loss')
plt.show()
```

---

➡️ **Next / 下一步**: File 9 of 11

---

### Bokeh Lines

# 25 — Bokeh Lines / 25 Bokeh Lines

**Chapter 26 — File 9 of 11 / 第26章 — 第9个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Load dataset**.

本脚本演示 **Load dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
```

---
## Step 1 — Step 1

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show
```

---
## Step 2 — Load dataset

```python
(x_train, train_labels), (_, _) = mnist.load_data()
```

---
## Step 3 — Choose only the digits 0, 1, 2

```python
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
```

---
## Step 4 — Verify the shape of training data

```python
total_examples, img_length, img_width = x_train.shape
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)
```

---
## Step 5 — Prepare for classifier network

```python
epochs = 10
y_train = utils.to_categorical(train_labels)
input_dim = img_length*img_width
```

---
## Step 6 — Create a Sequential model

```python
model = Sequential()
```

---
## Step 7 — First layer for reshaping input images from 2D to 1D

```python
model.add(Reshape((input_dim, ), input_shape=(img_length, img_width)))
```

---
## Step 8 — Dense layer of 8 neurons

```python
model.add(Dense(8, activation='relu'))
```

---
## Step 9 — Output layer

```python
model.add(Dense(total_classes, activation='softmax'))
```

---
## Step 10 — Compile model

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.33,
                    epochs=epochs, batch_size=10, verbose=0)
```

---
## Step 11 — Prepare pandas DataFrame

```python
df_history = pd.DataFrame(history.history)
print(df_history)
```

---
## Step 12 — Plot accuracy in Bokeh

```python
p = figure(title="Training and validation accuracy",
           x_axis_label="Epochs", y_axis_label="Accuracy")
epochs_array = np.arange(epochs)
p.line(epochs_array, df_history['accuracy'], legend_label="Training",
       color="blue", line_width=2)
p.line(epochs_array, df_history['val_accuracy'], legend_label="Validation",
       color="green")
p.legend.click_policy = "hide"
p.legend.location = 'bottom_right'
show(p)
```

---
## Learning Notes / 学习笔记

- **概念**: Load dataset 是机器学习中的常用技术。  
  *Load dataset is a common technique in machine learning.*

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
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Bokeh Lines / 25 Bokeh Lines
# Complete Code / 完整代码
# ===============================

from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show

# Load dataset
(x_train, train_labels), (_, _) = mnist.load_data()
# Choose only the digits 0, 1, 2
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
# Verify the shape of training data
total_examples, img_length, img_width = x_train.shape
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)

# Prepare for classifier network
epochs = 10
y_train = utils.to_categorical(train_labels)
input_dim = img_length*img_width
# Create a Sequential model
model = Sequential()
# First layer for reshaping input images from 2D to 1D
model.add(Reshape((input_dim, ), input_shape=(img_length, img_width)))
# Dense layer of 8 neurons
model.add(Dense(8, activation='relu'))
# Output layer
model.add(Dense(total_classes, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.33,
                    epochs=epochs, batch_size=10, verbose=0)

# Prepare pandas DataFrame
df_history = pd.DataFrame(history.history)
print(df_history)

# Plot accuracy in Bokeh
p = figure(title="Training and validation accuracy",
           x_axis_label="Epochs", y_axis_label="Accuracy")
epochs_array = np.arange(epochs)
p.line(epochs_array, df_history['accuracy'], legend_label="Training",
       color="blue", line_width=2)
p.line(epochs_array, df_history['val_accuracy'], legend_label="Validation",
       color="green")
p.legend.click_policy = "hide"
p.legend.location = 'bottom_right'
show(p)
```

---

➡️ **Next / 下一步**: File 10 of 11

---

### Sidebyside

# 26 — Sidebyside / 26 Sidebyside

**Chapter 26 — File 10 of 11 / 第26章 — 第10个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Load dataset**.

本脚本演示 **Load dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
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
## Step 1 — Step 1

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow import dtypes, tensordot
from tensorflow import convert_to_tensor, linalg, transpose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

---
## Step 2 — Load dataset

```python
(x_train, train_labels), (_, _) = mnist.load_data()
```

---
## Step 3 — Choose only the digits 0, 1, 2

```python
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
```

---
## Step 4 — Verify the shape of training data

```python
total_examples, img_length, img_width = x_train.shape
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)
```

---
## Step 5 — Convert the dataset into a 2D array of shape 18623 x 784

```python
x = convert_to_tensor(np.reshape(x_train, (x_train.shape[0], -1)),
                      dtype=dtypes.float32)
```

---
## Step 6 — Eigen-decomposition from a 784 x 784 matrix

```python
eigenvalues, eigenvectors = linalg.eigh(tensordot(transpose(x), x, axes=1))
```

---
## Step 7 — Print the three largest eigenvalues

```python
print('3 largest eigenvalues: ', eigenvalues[-3:])
```

---
## Step 8 — Project the data to eigenvectors

```python
x_pca = tensordot(x, eigenvectors, axes=1)
```

---
## Step 9 — Prepare for classifier network

```python
epochs = 10
y_train = utils.to_categorical(train_labels)
input_dim = img_length*img_width
```

---
## Step 10 — Create a Sequential model

```python
model = Sequential()
```

---
## Step 11 — First layer for reshaping input images from 2D to 1D

```python
model.add(Reshape((input_dim, ), input_shape=(img_length, img_width)))
```

---
## Step 12 — Dense layer of 8 neurons

```python
model.add(Dense(8, activation='relu'))
```

---
## Step 13 — Output layer

```python
model.add(Dense(total_classes, activation='softmax'))
```

---
## Step 14 — Compile model

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.33,
                    epochs=epochs, batch_size=10, verbose=0)
```

---
## Step 15 — Prepare pandas DataFrame

```python
df_history = pd.DataFrame(history.history)
print(df_history)
```

---
## Step 16 — Plot side-by-side

```python
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
```

---
## Step 17 — left plot

```python
scatter = ax[0].scatter(x_pca[:, -1], x_pca[:, -2], c=train_labels, s=5)
legend_plt = ax[0].legend(*scatter.legend_elements(),
                         loc="lower left", title="Digits")
ax[0].add_artist(legend_plt)
ax[0].set_title('First Two Dimensions of Projected Data After Applying PCA')
```

---
## Step 18 — right plot

```python
my_plot = sns.lineplot(data=df_history[["loss","val_loss"]], ax=ax[1])
my_plot.set_xlabel('Epochs')
my_plot.set_ylabel('Loss')
ax[1].legend(labels=["Training", "Validation"])
ax[1].set_title('Training and Validation Loss')
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Load dataset 是机器学习中的常用技术。  
  *Load dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `PCA` | 主成分分析，降维 | Principal Component Analysis, dimensionality reduction |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Sidebyside / 26 Sidebyside
# Complete Code / 完整代码
# ===============================

from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow import dtypes, tensordot
from tensorflow import convert_to_tensor, linalg, transpose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
(x_train, train_labels), (_, _) = mnist.load_data()
# Choose only the digits 0, 1, 2
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
# Verify the shape of training data
total_examples, img_length, img_width = x_train.shape
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)


# Convert the dataset into a 2D array of shape 18623 x 784
x = convert_to_tensor(np.reshape(x_train, (x_train.shape[0], -1)),
                      dtype=dtypes.float32)
# Eigen-decomposition from a 784 x 784 matrix
eigenvalues, eigenvectors = linalg.eigh(tensordot(transpose(x), x, axes=1))
# Print the three largest eigenvalues
print('3 largest eigenvalues: ', eigenvalues[-3:])
# Project the data to eigenvectors
x_pca = tensordot(x, eigenvectors, axes=1)


# Prepare for classifier network
epochs = 10
y_train = utils.to_categorical(train_labels)
input_dim = img_length*img_width
# Create a Sequential model
model = Sequential()
# First layer for reshaping input images from 2D to 1D
model.add(Reshape((input_dim, ), input_shape=(img_length, img_width)))
# Dense layer of 8 neurons
model.add(Dense(8, activation='relu'))
# Output layer
model.add(Dense(total_classes, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.33,
                    epochs=epochs, batch_size=10, verbose=0)


# Prepare pandas DataFrame
df_history = pd.DataFrame(history.history)
print(df_history)


# Plot side-by-side
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
# left plot
scatter = ax[0].scatter(x_pca[:, -1], x_pca[:, -2], c=train_labels, s=5)
legend_plt = ax[0].legend(*scatter.legend_elements(),
                         loc="lower left", title="Digits")
ax[0].add_artist(legend_plt)
ax[0].set_title('First Two Dimensions of Projected Data After Applying PCA')
# right plot
my_plot = sns.lineplot(data=df_history[["loss","val_loss"]], ax=ax[1])
my_plot.set_xlabel('Epochs')
my_plot.set_ylabel('Loss')
ax[1].legend(labels=["Training", "Validation"])
ax[1].set_title('Training and Validation Loss')
plt.show()
```

---

➡️ **Next / 下一步**: File 11 of 11

---

### Sidebyside

# 27 — Sidebyside / 27 Sidebyside

**Chapter 26 — File 11 of 11 / 第26章 — 第11个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Load dataset**.

本脚本演示 **Load dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
```

---
## Step 1 — Step 1

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow import dtypes, tensordot
from tensorflow import convert_to_tensor, linalg, transpose
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.layouts import row
```

---
## Step 2 — Load dataset

```python
(x_train, train_labels), (_, _) = mnist.load_data()
```

---
## Step 3 — Choose only the digits 0, 1, 2

```python
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
```

---
## Step 4 — Verify the shape of training data

```python
total_examples, img_length, img_width = x_train.shape
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)
```

---
## Step 5 — Convert the dataset into a 2D array of shape 18623 x 784

```python
x = convert_to_tensor(np.reshape(x_train, (x_train.shape[0], -1)),
                      dtype=dtypes.float32)
```

---
## Step 6 — Eigen-decomposition from a 784 x 784 matrix

```python
eigenvalues, eigenvectors = linalg.eigh(tensordot(transpose(x), x, axes=1))
```

---
## Step 7 — Print the three largest eigenvalues

```python
print('3 largest eigenvalues: ', eigenvalues[-3:])
```

---
## Step 8 — Project the data to eigenvectors

```python
x_pca = tensordot(x, eigenvectors, axes=1)
```

---
## Step 9 — Prepare for classifier network

```python
epochs = 10
y_train = utils.to_categorical(train_labels)
input_dim = img_length*img_width
```

---
## Step 10 — Create a Sequential model

```python
model = Sequential()
```

---
## Step 11 — First layer for reshaping input images from 2D to 1D

```python
model.add(Reshape((input_dim, ), input_shape=(img_length, img_width)))
```

---
## Step 12 — Dense layer of 8 neurons

```python
model.add(Dense(8, activation='relu'))
```

---
## Step 13 — Output layer

```python
model.add(Dense(total_classes, activation='softmax'))
```

---
## Step 14 — Compile model

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.33,
                    epochs=epochs, batch_size=10, verbose=0)
```

---
## Step 15 — Prepare pandas DataFrame

```python
df_history = pd.DataFrame(history.history)
print(df_history)
```

---
## Step 16 — Create scatter plot in Bokeh

```python
colormap = {0: "red", 1:"green", 2:"blue"}
my_scatter = figure(title="First Two Dimensions of Projected Data After Applying PCA",
                    x_axis_label="Dimension 1",
                    y_axis_label="Dimension 2",
                    width=500, height=400)
for digit in [0, 1, 2]:
    selection = x_pca[train_labels == digit]
    my_scatter.scatter(selection[:,-1].numpy(), selection[:,-2].numpy(),
                       color=colormap[digit], size=5, alpha=0.5,
                       legend_label="Digit "+str(digit))
my_scatter.legend.click_policy = "hide"
```

---
## Step 17 — Plot accuracy in Bokeh

```python
p = figure(title="Training and validation accuracy",
           x_axis_label="Epochs", y_axis_label="Accuracy",
           width=500, height=400)
epochs_array = np.arange(epochs)
p.line(epochs_array, df_history['accuracy'], legend_label="Training",
       color="blue", line_width=2)
p.line(epochs_array, df_history['val_accuracy'], legend_label="Validation",
       color="green")
p.legend.click_policy = "hide"
p.legend.location = 'bottom_right'

show(row(my_scatter, p))
```

---
## Learning Notes / 学习笔记

- **概念**: Load dataset 是机器学习中的常用技术。  
  *Load dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `PCA` | 主成分分析，降维 | Principal Component Analysis, dimensionality reduction |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Sidebyside / 27 Sidebyside
# Complete Code / 完整代码
# ===============================

from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow import dtypes, tensordot
from tensorflow import convert_to_tensor, linalg, transpose
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.layouts import row

# Load dataset
(x_train, train_labels), (_, _) = mnist.load_data()
# Choose only the digits 0, 1, 2
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
# Verify the shape of training data
total_examples, img_length, img_width = x_train.shape
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)


# Convert the dataset into a 2D array of shape 18623 x 784
x = convert_to_tensor(np.reshape(x_train, (x_train.shape[0], -1)),
                      dtype=dtypes.float32)
# Eigen-decomposition from a 784 x 784 matrix
eigenvalues, eigenvectors = linalg.eigh(tensordot(transpose(x), x, axes=1))
# Print the three largest eigenvalues
print('3 largest eigenvalues: ', eigenvalues[-3:])
# Project the data to eigenvectors
x_pca = tensordot(x, eigenvectors, axes=1)


# Prepare for classifier network
epochs = 10
y_train = utils.to_categorical(train_labels)
input_dim = img_length*img_width
# Create a Sequential model
model = Sequential()
# First layer for reshaping input images from 2D to 1D
model.add(Reshape((input_dim, ), input_shape=(img_length, img_width)))
# Dense layer of 8 neurons
model.add(Dense(8, activation='relu'))
# Output layer
model.add(Dense(total_classes, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.33,
                    epochs=epochs, batch_size=10, verbose=0)


# Prepare pandas DataFrame
df_history = pd.DataFrame(history.history)
print(df_history)


# Create scatter plot in Bokeh
colormap = {0: "red", 1:"green", 2:"blue"}
my_scatter = figure(title="First Two Dimensions of Projected Data After Applying PCA",
                    x_axis_label="Dimension 1",
                    y_axis_label="Dimension 2",
                    width=500, height=400)
for digit in [0, 1, 2]:
    selection = x_pca[train_labels == digit]
    my_scatter.scatter(selection[:,-1].numpy(), selection[:,-2].numpy(),
                       color=colormap[digit], size=5, alpha=0.5,
                       legend_label="Digit "+str(digit))
my_scatter.legend.click_policy = "hide"


# Plot accuracy in Bokeh
p = figure(title="Training and validation accuracy",
           x_axis_label="Epochs", y_axis_label="Accuracy",
           width=500, height=400)
epochs_array = np.arange(epochs)
p.line(epochs_array, df_history['accuracy'], legend_label="Training",
       color="blue", line_width=2)
p.line(epochs_array, df_history['val_accuracy'], legend_label="Validation",
       color="green")
p.legend.click_policy = "hide"
p.legend.location = 'bottom_right'

show(row(my_scatter, p))
```

---

### Chapter Summary

# Chapter 26 Summary / 第26章总结

## Theme / 主题: Chapter 26 / Chapter 26

This chapter contains **11 code files** demonstrating chapter 26.

本章包含 **11 个代码文件**，演示Chapter 26。

---
## Evolution / 演化路线

  1. `03_imshow.ipynb` — Imshow
  2. `04_eigen.ipynb` — Eigen
  3. `06_scatter.ipynb` — Scatter
  4. `09_3d_scatter.ipynb` — 3D Scatter
  5. `14_seaborn_scatter.ipynb` — Seaborn Scatter
  6. `16_bokeh_scatter.ipynb` — Bokeh Scatter
  7. `21_history_lines.ipynb` — History Lines
  8. `23_seaborn_lines.ipynb` — Seaborn Lines
  9. `25_bokeh_lines.ipynb` — Bokeh Lines
  10. `26_sidebyside.ipynb` — Sidebyside
  11. `27_sidebyside.ipynb` — Sidebyside

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 26) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 26）是机器学习流水线中的基础构建块。

---
