# 注意力与Transformer / Transformer Models with Attention
## Chapter 07

---

### Weights

# 03 — Weights / 03 Weights

**Chapter 07 — File 1 of 3 / 第07章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Weights**.

本脚本演示 **03 Weights**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  ⚙️ 配置训练 / Configure Training
```

---
## Step 1 — Step 1

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, SimpleRNN

def create_RNN(hidden_units, dense_units, input_shape, activation):
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(SimpleRNN(hidden_units, input_shape=input_shape,
                        activation=activation[0]))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(units=dense_units, activation=activation[1]))
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

demo_model = create_RNN(2, 1, (3,1), activation=['linear', 'linear'])

wx = demo_model.get_weights()[0]
wh = demo_model.get_weights()[1]
bh = demo_model.get_weights()[2]
wy = demo_model.get_weights()[3]
by = demo_model.get_weights()[4]

# 打印输出 / Print output
print('wx = ', wx, ' wh = ', wh, ' bh = ', bh, ' wy =', wy, 'by = ', by)
```

---
## Learning Notes / 学习笔记

- **概念**: Weights 是机器学习中的常用技术。  
  *Weights is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Weights / 03 Weights
# Complete Code / 完整代码
# ===============================

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, SimpleRNN

def create_RNN(hidden_units, dense_units, input_shape, activation):
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(SimpleRNN(hidden_units, input_shape=input_shape,
                        activation=activation[0]))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(units=dense_units, activation=activation[1]))
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

demo_model = create_RNN(2, 1, (3,1), activation=['linear', 'linear'])

wx = demo_model.get_weights()[0]
wh = demo_model.get_weights()[1]
bh = demo_model.get_weights()[2]
wy = demo_model.get_weights()[3]
by = demo_model.get_weights()[4]

# 打印输出 / Print output
print('wx = ', wx, ' wh = ', wh, ' bh = ', bh, ' wy =', wy, 'by = ', by)
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Threesteps



---

### Simplernn

# 10 — Simplernn / 循环神经网络

**Chapter 07 — File 3 of 3 / 第07章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Parameter split_percent defines the ratio of training examples**.

本脚本演示 **Parameter split_percent defines the ratio of training examples**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — Step 1

```python
# 导入数学函数库 / Import math functions library
import math

# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, SimpleRNN
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
```

---
## Step 2 — Parameter split_percent defines the ratio of training examples

```python
def get_train_test(url, split_percent=0.8):
    # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
    df = read_csv(url, usecols=[1], engine='python')
    # 转换为NumPy数组 / Convert to NumPy array
    data = np.array(df.values.astype('float32'))
    # 归一化到[0,1]范围 / Normalize to [0,1] range
    scaler = MinMaxScaler(feature_range=(0, 1))
    # 展平为一维数组 / Flatten to 1D array
    data = scaler.fit_transform(data).flatten()
    # 获取长度 / Get length
    n = len(data)
```

---
## Step 3 — Point for splitting data into train and test

```python
split = int(n*split_percent)
    # 生成整数序列 / Generate integer sequence
    train_data = data[range(split)]
    test_data = data[split:]
    return train_data, test_data, data
```

---
## Step 4 — Prepare the input X and target Y

```python
def get_XY(dat, time_steps):
    # 生成等差数组 / Generate array with step
    Y_ind = np.arange(time_steps, len(dat), time_steps)
    Y = dat[Y_ind]
    # 获取长度 / Get length
    rows_x = len(Y)
    # 生成整数序列 / Generate integer sequence
    X = dat[range(time_steps*rows_x)]
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    X = np.reshape(X, (rows_x, time_steps, 1))
    return X, Y

def create_RNN(hidden_units, dense_units, input_shape, activation):
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(units=dense_units, activation=activation[1]))
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def print_error(trainY, testY, train_predict, test_predict):
```

---
## Step 5 — Error of predictions

```python
# 计算均方误差 / Calculate Mean Squared Error
train_rmse = math.sqrt(mean_squared_error(trainY, train_predict))
    # 计算均方误差 / Calculate Mean Squared Error
    test_rmse = math.sqrt(mean_squared_error(testY, test_predict))
```

---
## Step 6 — Print RMSE

```python
# 打印输出 / Print output
print('Train RMSE: %.3f RMSE' % (train_rmse))
    # 打印输出 / Print output
    print('Test RMSE: %.3f RMSE' % (test_rmse))
```

---
## Step 7 — Plot the result

```python
def plot_result(trainY, testY, train_predict, test_predict):
    # 添加元素到列表末尾 / Append element to list end
    actual = np.append(trainY, testY)
    # 添加元素到列表末尾 / Append element to list end
    predictions = np.append(train_predict, test_predict)
    # 获取长度 / Get length
    rows = len(actual)
    # 创建画布 / Create figure canvas
    plt.figure(figsize=(15, 6), dpi=80)
    # 绘制折线图 / Draw line plot
    plt.plot(range(rows), actual)
    # 绘制折线图 / Draw line plot
    plt.plot(range(rows), predictions)
    # 获取长度 / Get length
    plt.axvline(x=len(trainY), color='r')
    # 显示图例 / Show legend
    plt.legend(['Actual', 'Predictions'])
    # 设置X轴标签 / Set X-axis label
    plt.xlabel('Observation number after given time steps')
    # 设置Y轴标签 / Set Y-axis label
    plt.ylabel('Sunspots scaled')
    # 设置图表标题 / Set chart title
    plt.title('Actual and Predicted Values')

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv"
time_steps = 12
train_data, test_data, data = get_train_test(url)
trainX, trainY = get_XY(train_data, time_steps)
testX, testY = get_XY(test_data, time_steps)
```

---
## Step 8 — Create model and train

```python
model = create_RNN(hidden_units=3, dense_units=1, input_shape=(time_steps,1),
                   activation=['tanh', 'tanh'])
# 训练模型 / Train the model
model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)
```

---
## Step 9 — make predictions

```python
# 用模型做预测 / Make predictions with model
train_predict = model.predict(trainX)
# 用模型做预测 / Make predictions with model
test_predict = model.predict(testX)
```

---
## Step 10 — Print error

```python
print_error(trainY, testY, train_predict, test_predict)
```

---
## Step 11 — Plot result

```python
plot_result(trainY, testY, train_predict, test_predict)
```

---
## Learning Notes / 学习笔记

- **概念**: Parameter split_percent defines the ratio of training examples 是机器学习中的常用技术。  
  *Parameter split_percent defines the ratio of training examples is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Simplernn / 循环神经网络
# Complete Code / 完整代码
# ===============================

# 导入数学函数库 / Import math functions library
import math

# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, SimpleRNN
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# Parameter split_percent defines the ratio of training examples
def get_train_test(url, split_percent=0.8):
    # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
    df = read_csv(url, usecols=[1], engine='python')
    # 转换为NumPy数组 / Convert to NumPy array
    data = np.array(df.values.astype('float32'))
    # 归一化到[0,1]范围 / Normalize to [0,1] range
    scaler = MinMaxScaler(feature_range=(0, 1))
    # 展平为一维数组 / Flatten to 1D array
    data = scaler.fit_transform(data).flatten()
    # 获取长度 / Get length
    n = len(data)
    # Point for splitting data into train and test
    split = int(n*split_percent)
    # 生成整数序列 / Generate integer sequence
    train_data = data[range(split)]
    test_data = data[split:]
    return train_data, test_data, data

# Prepare the input X and target Y
def get_XY(dat, time_steps):
    # 生成等差数组 / Generate array with step
    Y_ind = np.arange(time_steps, len(dat), time_steps)
    Y = dat[Y_ind]
    # 获取长度 / Get length
    rows_x = len(Y)
    # 生成整数序列 / Generate integer sequence
    X = dat[range(time_steps*rows_x)]
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    X = np.reshape(X, (rows_x, time_steps, 1))
    return X, Y

def create_RNN(hidden_units, dense_units, input_shape, activation):
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(units=dense_units, activation=activation[1]))
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def print_error(trainY, testY, train_predict, test_predict):
    # Error of predictions
    # 计算均方误差 / Calculate Mean Squared Error
    train_rmse = math.sqrt(mean_squared_error(trainY, train_predict))
    # 计算均方误差 / Calculate Mean Squared Error
    test_rmse = math.sqrt(mean_squared_error(testY, test_predict))
    # Print RMSE
    # 打印输出 / Print output
    print('Train RMSE: %.3f RMSE' % (train_rmse))
    # 打印输出 / Print output
    print('Test RMSE: %.3f RMSE' % (test_rmse))

# Plot the result
def plot_result(trainY, testY, train_predict, test_predict):
    # 添加元素到列表末尾 / Append element to list end
    actual = np.append(trainY, testY)
    # 添加元素到列表末尾 / Append element to list end
    predictions = np.append(train_predict, test_predict)
    # 获取长度 / Get length
    rows = len(actual)
    # 创建画布 / Create figure canvas
    plt.figure(figsize=(15, 6), dpi=80)
    # 绘制折线图 / Draw line plot
    plt.plot(range(rows), actual)
    # 绘制折线图 / Draw line plot
    plt.plot(range(rows), predictions)
    # 获取长度 / Get length
    plt.axvline(x=len(trainY), color='r')
    # 显示图例 / Show legend
    plt.legend(['Actual', 'Predictions'])
    # 设置X轴标签 / Set X-axis label
    plt.xlabel('Observation number after given time steps')
    # 设置Y轴标签 / Set Y-axis label
    plt.ylabel('Sunspots scaled')
    # 设置图表标题 / Set chart title
    plt.title('Actual and Predicted Values')

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv"
time_steps = 12
train_data, test_data, data = get_train_test(url)
trainX, trainY = get_XY(train_data, time_steps)
testX, testY = get_XY(test_data, time_steps)

# Create model and train
model = create_RNN(hidden_units=3, dense_units=1, input_shape=(time_steps,1),
                   activation=['tanh', 'tanh'])
# 训练模型 / Train the model
model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)

# make predictions
# 用模型做预测 / Make predictions with model
train_predict = model.predict(trainX)
# 用模型做预测 / Make predictions with model
test_predict = model.predict(testX)

# Print error
print_error(trainY, testY, train_predict, test_predict)

#Plot result
plot_result(trainY, testY, train_predict, test_predict)
```

---

### Chapter Summary / 章节总结

# Chapter 07 Summary / 第07章总结

## Theme / 主题: Chapter 07 / Chapter 07

This chapter contains **3 code files** demonstrating chapter 07.

本章包含 **3 个代码文件**，演示Chapter 07。

---
## Evolution / 演化路线

  1. `03_weights.ipynb` — Weights
  2. `04_threesteps.ipynb` — Threesteps
  3. `10_simplernn.ipynb` — Simplernn

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 07) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 07）是机器学习流水线中的基础构建块。

---
