# Python 机器学习 / Python for Machine Learning
## Chapter 25

---

### Sqlite3

# 08 — Sqlite3 / 08 Sqlite3

**Chapter 25 — File 1 of 11 / 第25章 — 第1个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Read dataset from OpenML**.

本脚本演示 **Read dataset from OpenML**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
import sqlite3

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_openml
```

---
## Step 2 — Read dataset from OpenML

```python
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
# 打印输出 / Print output
print("Data from OpenML:")
# 打印输出 / Print output
print(type(dataset))
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataset.head())
```

---
## Step 3 — Create database

```python
conn = sqlite3.connect(":memory:")
cur = conn.cursor()
create_sql = """
    CREATE TABLE diabetes(
        preg NUM,
        plas NUM,
        pres NUM,
        skin NUM,
        insu NUM,
        mass NUM,
        pedi NUM,
        age NUM,
        class TEXT
    )
"""
cur.execute(create_sql)
```

---
## Step 4 — Insert data into the table using a parameterized SQL

```python
insert_sql = "INSERT INTO diabetes VALUES (?,?,?,?,?,?,?,?,?)"
rows = dataset.to_numpy().tolist()
cur.executemany(insert_sql, rows)

def cursor2dataframe(cur):
    """Read the column header from the cursor and then the rows of
    data from it. Afterwards, create a DataFrame"""
    header = [x[0] for x in cur.description]
```

---
## Step 5 — gets data from the last executed SQL query

```python
data = cur.fetchall()
```

---
## Step 6 — convert the data into a pandas DataFrame

```python
return pd.DataFrame(data, columns=header)
```

---
## Step 7 — get 5 random rows from the diabetes table

```python
select_sql = "SELECT * FROM diabetes ORDER BY random() LIMIT 5"
cur.execute(select_sql)
sample = cursor2dataframe(cur)
# 打印输出 / Print output
print("Data from SQLite database:")
# 打印输出 / Print output
print(sample)
```

---
## Step 8 — close database connection

```python
conn.commit()
conn.close()
```

---
## Learning Notes / 学习笔记

- **概念**: Read dataset from OpenML 是机器学习中的常用技术。  
  *Read dataset from OpenML is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `head()` | 查看前几行数据 | View first few rows |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Sqlite3 / 08 Sqlite3
# Complete Code / 完整代码
# ===============================

import sqlite3

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_openml

# Read dataset from OpenML
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
# 打印输出 / Print output
print("Data from OpenML:")
# 打印输出 / Print output
print(type(dataset))
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataset.head())

# Create database
conn = sqlite3.connect(":memory:")
cur = conn.cursor()
create_sql = """
    CREATE TABLE diabetes(
        preg NUM,
        plas NUM,
        pres NUM,
        skin NUM,
        insu NUM,
        mass NUM,
        pedi NUM,
        age NUM,
        class TEXT
    )
"""
cur.execute(create_sql)

# Insert data into the table using a parameterized SQL
insert_sql = "INSERT INTO diabetes VALUES (?,?,?,?,?,?,?,?,?)"
rows = dataset.to_numpy().tolist()
cur.executemany(insert_sql, rows)

def cursor2dataframe(cur):
    """Read the column header from the cursor and then the rows of
    data from it. Afterwards, create a DataFrame"""
    header = [x[0] for x in cur.description]
    # gets data from the last executed SQL query
    data = cur.fetchall()
    # convert the data into a pandas DataFrame
    return pd.DataFrame(data, columns=header)

# get 5 random rows from the diabetes table
select_sql = "SELECT * FROM diabetes ORDER BY random() LIMIT 5"
cur.execute(select_sql)
sample = cursor2dataframe(cur)
# 打印输出 / Print output
print("Data from SQLite database:")
# 打印输出 / Print output
print(sample)

# close database connection
conn.commit()
conn.close()
```

---

➡️ **Next / 下一步**: File 2 of 11

---

### Generator



---

### Dbm

# 15 — Dbm / 15 Dbm

**Chapter 25 — File 3 of 11 / 第25章 — 第3个文件（共11个）**

---

## Summary / 总结

This script demonstrates **get digits dataset (8x8 images of digits)**.

本脚本演示 **get digits dataset (8x8 images of digits)**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — Step 1

```python
import dbm
# 导入对象序列化模块 / Import object serialization module
import pickle
# 导入随机数生成模块 / Import random number module
import random

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
import sklearn.datasets
```

---
## Step 2 — get digits dataset (8x8 images of digits)

```python
digits = sklearn.datasets.load_digits()
```

---
## Step 3 — create file if not exists, otherwise open for read/write

```python
with dbm.open("digits.dbm", "c") as db:
    # 获取长度 / Get length
    for idx in range(len(digits.target)):
        db[str(idx)] = pickle.dumps((digits.images[idx], digits.target[idx]))
```

---
## Step 4 — number of images that we want in our sample

```python
batchsize = 4
images = []
targets = []
```

---
## Step 5 — open the database and read a sample

```python
with dbm.open("digits.dbm", "r") as db:
```

---
## Step 6 — get all keys from the database

```python
# 获取字典的所有键 / Get all dict keys
keys = db.keys()
```

---
## Step 7 — randomly samples n keys

```python
for key in random.sample(keys, batchsize):
```

---
## Step 8 — go through each key in the random sample

```python
image, target = pickle.loads(db[key])
        # 添加元素到列表末尾 / Append element to list end
        images.append(image)
        # 添加元素到列表末尾 / Append element to list end
        targets.append(target)
    # 创建NumPy数组 / Create NumPy array
    print(np.array(images), np.array(targets))
```

---
## Learning Notes / 学习笔记

- **概念**: get digits dataset (8x8 images of digits) 是机器学习中的常用技术。  
  *get digits dataset (8x8 images of digits) is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Dbm / 15 Dbm
# Complete Code / 完整代码
# ===============================

import dbm
# 导入对象序列化模块 / Import object serialization module
import pickle
# 导入随机数生成模块 / Import random number module
import random

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
import sklearn.datasets

# get digits dataset (8x8 images of digits)
digits = sklearn.datasets.load_digits()

# create file if not exists, otherwise open for read/write
with dbm.open("digits.dbm", "c") as db:
    # 获取长度 / Get length
    for idx in range(len(digits.target)):
        db[str(idx)] = pickle.dumps((digits.images[idx], digits.target[idx]))

# number of images that we want in our sample
batchsize = 4
images = []
targets = []

# open the database and read a sample
with dbm.open("digits.dbm", "r") as db:
    # get all keys from the database
    # 获取字典的所有键 / Get all dict keys
    keys = db.keys()
    # randomly samples n keys
    for key in random.sample(keys, batchsize):
        # go through each key in the random sample
        image, target = pickle.loads(db[key])
        # 添加元素到列表末尾 / Append element to list end
        images.append(image)
        # 添加元素到列表末尾 / Append element to list end
        targets.append(target)
    # 创建NumPy数组 / Create NumPy array
    print(np.array(images), np.array(targets))
```

---

➡️ **Next / 下一步**: File 4 of 11

---

### Digits

# 18 — Digits / 18 Digits

**Chapter 25 — File 4 of 11 / 第25章 — 第4个文件（共11个）**

---

## Summary / 总结

This script demonstrates **get digits dataset (8x8 images of digits)**.

本脚本演示 **get digits dataset (8x8 images of digits)**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  🏋️ 训练模型 / Train Model
       │
       ▼
  📊 评估模型 / Evaluate Model
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — Step 1

```python
import dbm
# 导入对象序列化模块 / Import object serialization module
import pickle
# 导入随机数生成模块 / Import random number module
import random

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
import sklearn.datasets
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
```

---
## Step 2 — get digits dataset (8x8 images of digits)

```python
digits = sklearn.datasets.load_digits()
```

---
## Step 3 — create file if not exists, otherwise open for read/write

```python
with dbm.open("digits.dbm", "c") as db:
    # 获取长度 / Get length
    for idx in range(len(digits.target)):
        db[str(idx)] = pickle.dumps((digits.images[idx], digits.target[idx]))
```

---
## Step 4 — retrieving data from database for model

```python
def datagen(batch_size):
    """A generator to produce samples from database
    """
    with dbm.open("digits.dbm", "r") as db:
        # 获取字典的所有键 / Get all dict keys
        keys = db.keys()
        while True:
            images = []
            targets = []
            for key in random.sample(keys, batch_size):
                image, target = pickle.loads(db[key])
                # 添加元素到列表末尾 / Append element to list end
                images.append(image)
                # 添加元素到列表末尾 / Append element to list end
                targets.append(target)
            # 改变数组形状（不改变数据） / Reshape array (data unchanged)
            yield np.array(images).reshape(-1,64), np.array(targets)
```

---
## Step 5 — Classification model in Keras

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(32, input_dim=64, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(32, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(10, activation='softmax'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["sparse_categorical_accuracy"])
```

---
## Step 6 — Train with data from dbm store

```python
# 训练模型 / Train the model
history = model.fit(datagen(32), epochs=5, steps_per_epoch=1000)
```

---
## Learning Notes / 学习笔记

- **概念**: get digits dataset (8x8 images of digits) 是机器学习中的常用技术。  
  *get digits dataset (8x8 images of digits) is a common technique in machine learning.*

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
| `model.fit` | 训练模型 | Train the model |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Digits / 18 Digits
# Complete Code / 完整代码
# ===============================

import dbm
# 导入对象序列化模块 / Import object serialization module
import pickle
# 导入随机数生成模块 / Import random number module
import random

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
import sklearn.datasets
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense

# get digits dataset (8x8 images of digits)
digits = sklearn.datasets.load_digits()

# create file if not exists, otherwise open for read/write
with dbm.open("digits.dbm", "c") as db:
    # 获取长度 / Get length
    for idx in range(len(digits.target)):
        db[str(idx)] = pickle.dumps((digits.images[idx], digits.target[idx]))

# retrieving data from database for model
def datagen(batch_size):
    """A generator to produce samples from database
    """
    with dbm.open("digits.dbm", "r") as db:
        # 获取字典的所有键 / Get all dict keys
        keys = db.keys()
        while True:
            images = []
            targets = []
            for key in random.sample(keys, batch_size):
                image, target = pickle.loads(db[key])
                # 添加元素到列表末尾 / Append element to list end
                images.append(image)
                # 添加元素到列表末尾 / Append element to list end
                targets.append(target)
            # 改变数组形状（不改变数据） / Reshape array (data unchanged)
            yield np.array(images).reshape(-1,64), np.array(targets)

# Classification model in Keras
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(32, input_dim=64, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(32, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(10, activation='softmax'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["sparse_categorical_accuracy"])

# Train with data from dbm store
# 训练模型 / Train the model
history = model.fit(datagen(32), epochs=5, steps_per_epoch=1000)
```

---

➡️ **Next / 下一步**: File 5 of 11

---

### Excel

# 19 — Excel / 19 Excel

**Chapter 25 — File 5 of 11 / 第25章 — 第5个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Read dataset from OpenML**.

本脚本演示 **Read dataset from OpenML**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_openml
import openpyxl
```

---
## Step 2 — Read dataset from OpenML

```python
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
# 获取列名 / Get column names
header = list(dataset.columns)
data = dataset.to_numpy().tolist()
```

---
## Step 3 — Create Excel workbook and write data into the default worksheet

```python
wb = openpyxl.Workbook()
sheet = wb.active # use the default worksheet
sheet.title = "Diabetes"
# 同时获取索引和值 / Get both index and value
for n,colname in enumerate(header):
    sheet.cell(row=1, column=1+n, value=colname)
# 同时获取索引和值 / Get both index and value
for n,row in enumerate(data):
    # 同时获取索引和值 / Get both index and value
    for m,cell in enumerate(row):
        sheet.cell(row=2+n, column=1+m, value=cell)
```

---
## Step 4 — Save

```python
wb.save("MLM.xlsx")
```

---
## Learning Notes / 学习笔记

- **概念**: Read dataset from OpenML 是机器学习中的常用技术。  
  *Read dataset from OpenML is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Excel / 19 Excel
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_openml
import openpyxl

# Read dataset from OpenML
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
# 获取列名 / Get column names
header = list(dataset.columns)
data = dataset.to_numpy().tolist()

# Create Excel workbook and write data into the default worksheet
wb = openpyxl.Workbook()
sheet = wb.active # use the default worksheet
sheet.title = "Diabetes"
# 同时获取索引和值 / Get both index and value
for n,colname in enumerate(header):
    sheet.cell(row=1, column=1+n, value=colname)
# 同时获取索引和值 / Get both index and value
for n,row in enumerate(data):
    # 同时获取索引和值 / Get both index and value
    for m,cell in enumerate(row):
        sheet.cell(row=2+n, column=1+m, value=cell)
# Save
wb.save("MLM.xlsx")
```

---

➡️ **Next / 下一步**: File 6 of 11

---

### Excel

# 22 — Excel / 22 Excel

**Chapter 25 — File 6 of 11 / 第25章 — 第6个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Read dataset from OpenML**.

本脚本演示 **Read dataset from OpenML**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_openml
import openpyxl
```

---
## Step 2 — Read dataset from OpenML

```python
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
# 获取列名 / Get column names
header = list(dataset.columns)
data = dataset.to_numpy().tolist()
```

---
## Step 3 — Create Excel workbook and write data into the default worksheet

```python
wb = openpyxl.Workbook()
sheet = wb.create_sheet("Diabetes")  # or wb.active for default sheet
# 添加元素到列表末尾 / Append element to list end
sheet.append(header)
for row in data:
    # 添加元素到列表末尾 / Append element to list end
    sheet.append(row)
```

---
## Step 4 — Save

```python
wb.save("MLM.xlsx")
```

---
## Learning Notes / 学习笔记

- **概念**: Read dataset from OpenML 是机器学习中的常用技术。  
  *Read dataset from OpenML is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Excel / 22 Excel
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_openml
import openpyxl

# Read dataset from OpenML
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
# 获取列名 / Get column names
header = list(dataset.columns)
data = dataset.to_numpy().tolist()

# Create Excel workbook and write data into the default worksheet
wb = openpyxl.Workbook()
sheet = wb.create_sheet("Diabetes")  # or wb.active for default sheet
# 添加元素到列表末尾 / Append element to list end
sheet.append(header)
for row in data:
    # 添加元素到列表末尾 / Append element to list end
    sheet.append(row)
# Save
wb.save("MLM.xlsx")
```

---

➡️ **Next / 下一步**: File 7 of 11

---

### Excelkeras



---

### Inmemory

# 24 — Inmemory / 24 Inmemory

**Chapter 25 — File 8 of 11 / 第25章 — 第8个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Read data from OpenML**.

本脚本演示 **Read data from OpenML**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  🏋️ 训练模型 / Train Model
       │
       ▼
  📊 评估模型 / Evaluate Model
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — Step 1

```python
# 导入随机数生成模块 / Import random number module
import random

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
import openpyxl
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_openml
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
```

---
## Step 2 — Read data from OpenML

```python
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
# 获取列名 / Get column names
header = list(dataset.columns)
rows = dataset.to_numpy().tolist()
```

---
## Step 3 — Create Excel workbook and write data into the default worksheet

```python
wb = openpyxl.Workbook()
sheet = wb.active
sheet.title = "Diabetes"
# 添加元素到列表末尾 / Append element to list end
sheet.append(header)
for row in rows:
    # 添加元素到列表末尾 / Append element to list end
    sheet.append(row)
```

---
## Step 4 — Save

```python
wb.save("MLM.xlsx")
```

---
## Step 5 — Read entire worksheet from the Excel file

```python
wb = openpyxl.load_workbook("MLM.xlsx", read_only=True)
sheet = wb.active
X = []
y = []
# 同时获取索引和值 / Get both index and value
for i, row in enumerate(sheet.rows):
    if i==0:
        continue # skip the header row
    rowdata = [cell.value for cell in row]
    # 添加元素到列表末尾 / Append element to list end
    X.append(rowdata[:-1])
    # 添加元素到列表末尾 / Append element to list end
    y.append(1 if rowdata[-1]=="tested_positive" else 0)
X, y = np.asarray(X), np.asarray(y)
```

---
## Step 6 — create binary classification model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(16, input_dim=8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 7 — train model

```python
# 训练模型 / Train the model
history = model.fit(X, y, epochs=5)
```

---
## Learning Notes / 学习笔记

- **概念**: Read data from OpenML 是机器学习中的常用技术。  
  *Read data from OpenML is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Inmemory / 24 Inmemory
# Complete Code / 完整代码
# ===============================

# 导入随机数生成模块 / Import random number module
import random

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
import openpyxl
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_openml
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense

# Read data from OpenML
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
# 获取列名 / Get column names
header = list(dataset.columns)
rows = dataset.to_numpy().tolist()

# Create Excel workbook and write data into the default worksheet
wb = openpyxl.Workbook()
sheet = wb.active
sheet.title = "Diabetes"
# 添加元素到列表末尾 / Append element to list end
sheet.append(header)
for row in rows:
    # 添加元素到列表末尾 / Append element to list end
    sheet.append(row)
# Save
wb.save("MLM.xlsx")

# Read entire worksheet from the Excel file
wb = openpyxl.load_workbook("MLM.xlsx", read_only=True)
sheet = wb.active
X = []
y = []
# 同时获取索引和值 / Get both index and value
for i, row in enumerate(sheet.rows):
    if i==0:
        continue # skip the header row
    rowdata = [cell.value for cell in row]
    # 添加元素到列表末尾 / Append element to list end
    X.append(rowdata[:-1])
    # 添加元素到列表末尾 / Append element to list end
    y.append(1 if rowdata[-1]=="tested_positive" else 0)
X, y = np.asarray(X), np.asarray(y)

# create binary classification model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(16, input_dim=8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
# 训练模型 / Train the model
history = model.fit(X, y, epochs=5)
```

---

➡️ **Next / 下一步**: File 9 of 11

---

### Googlesheet



---

### Gspread

# 34 — Gspread / 34 Gspread

**Chapter 25 — File 10 of 11 / 第25章 — 第10个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Google Sheet ID, as granted access to the service account**.

本脚本演示 **Google Sheet ID, as granted access to the service account**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入随机数生成模块 / Import random number module
import random

import gspread
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_openml
```

---
## Step 2 — Google Sheet ID, as granted access to the service account

```python
sheet_id = '12Pc2_pX3HOSltcRLHtqiq3RSOL9RcG72CZxRqsMeRul'
```

---
## Step 3 — Connect to Google Sheet

```python
cred_file = "mlm-python.json"
gc = gspread.service_account(filename=cred_file)
sheet = gc.open_by_key(sheet_id)
spreadsheet = sheet.get_worksheet(0)
```

---
## Step 4 — Clear all data

```python
spreadsheet.clear()
```

---
## Step 5 — Read dataset from OpenML

```python
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
# 获取列名 / Get column names
rows = [list(dataset.columns)]       # column headers
rows += dataset.to_numpy().tolist()  # rows of data
```

---
## Step 6 — Write to spreadsheet

```python
spreadsheet.append_rows(rows)
```

---
## Step 7 — Check the number of rows and columns in the spreadsheet

```python
# 打印输出 / Print output
print(spreadsheet.row_count, spreadsheet.col_count)
```

---
## Step 8 — Read a random row of data

```python
maxcol = chr(ord("A") - 1 + spreadsheet.col_count)
row = random.randint(2, spreadsheet.row_count)
readrange = f"A{row}:{maxcol}{row}"
data = spreadsheet.get(readrange)
# 打印输出 / Print output
print(data)
```

---
## Learning Notes / 学习笔记

- **概念**: Google Sheet ID, as granted access to the service account 是机器学习中的常用技术。  
  *Google Sheet ID, as granted access to the service account is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Gspread / 34 Gspread
# Complete Code / 完整代码
# ===============================

# 导入随机数生成模块 / Import random number module
import random

import gspread
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_openml

# Google Sheet ID, as granted access to the service account
sheet_id = '12Pc2_pX3HOSltcRLHtqiq3RSOL9RcG72CZxRqsMeRul'

# Connect to Google Sheet
cred_file = "mlm-python.json"
gc = gspread.service_account(filename=cred_file)
sheet = gc.open_by_key(sheet_id)
spreadsheet = sheet.get_worksheet(0)

# Clear all data
spreadsheet.clear()

# Read dataset from OpenML
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
# 获取列名 / Get column names
rows = [list(dataset.columns)]       # column headers
rows += dataset.to_numpy().tolist()  # rows of data

# Write to spreadsheet
spreadsheet.append_rows(rows)

# Check the number of rows and columns in the spreadsheet
# 打印输出 / Print output
print(spreadsheet.row_count, spreadsheet.col_count)

# Read a random row of data
maxcol = chr(ord("A") - 1 + spreadsheet.col_count)
row = random.randint(2, spreadsheet.row_count)
readrange = f"A{row}:{maxcol}{row}"
data = spreadsheet.get(readrange)
# 打印输出 / Print output
print(data)
```

---

➡️ **Next / 下一步**: File 11 of 11

---

### Gspreadkeras

# 35 — Gspreadkeras / Keras

**Chapter 25 — File 11 of 11 / 第25章 — 第11个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Google Sheet ID, as granted access to the service account**.

本脚本演示 **Google Sheet ID, as granted access to the service account**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  🏋️ 训练模型 / Train Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Step 1

```python
# 导入随机数生成模块 / Import random number module
import random

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
import gspread
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_openml
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
```

---
## Step 2 — Google Sheet ID, as granted access to the service account

```python
sheet_id = '12Pc2_pX3HOSltcRLHtqiq3RSOL9RcG72CZxRqsMeRul'
```

---
## Step 3 — Connect to Google Sheet

```python
cred_file = "mlm-python.json"
gc = gspread.service_account(filename=cred_file)
sheet = gc.open_by_key(sheet_id)
spreadsheet = sheet.get_worksheet(0)
```

---
## Step 4 — Clear all data

```python
spreadsheet.clear()
```

---
## Step 5 — Read dataset from OpenML

```python
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
# 获取列名 / Get column names
rows = [list(dataset.columns)]       # column headers
rows += dataset.to_numpy().tolist()  # rows of data
```

---
## Step 6 — Write to spreadsheet

```python
spreadsheet.append_rows(rows)
```

---
## Step 7 — Read the entire spreadsheet, except header

```python
maxrow = spreadsheet.row_count
maxcol = chr(ord("A") - 1 + spreadsheet.col_count)
data = spreadsheet.get(f"A2:{maxcol}{maxrow}")
X = [row[:-1] for row in data]
y = [1 if row[-1]=="tested_positive" else 0 for row in data]
# 转换数据类型 / Convert data type
X, y = np.asarray(X).astype(float), np.asarray(y)
```

---
## Step 8 — create binary classification model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(16, input_dim=8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 9 — train model

```python
# 训练模型 / Train the model
history = model.fit(X, y, epochs=5)
```

---
## Learning Notes / 学习笔记

- **概念**: Google Sheet ID, as granted access to the service account 是机器学习中的常用技术。  
  *Google Sheet ID, as granted access to the service account is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Gspreadkeras / Keras
# Complete Code / 完整代码
# ===============================

# 导入随机数生成模块 / Import random number module
import random

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
import gspread
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_openml
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense

# Google Sheet ID, as granted access to the service account
sheet_id = '12Pc2_pX3HOSltcRLHtqiq3RSOL9RcG72CZxRqsMeRul'

# Connect to Google Sheet
cred_file = "mlm-python.json"
gc = gspread.service_account(filename=cred_file)
sheet = gc.open_by_key(sheet_id)
spreadsheet = sheet.get_worksheet(0)

# Clear all data
spreadsheet.clear()

# Read dataset from OpenML
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
# 获取列名 / Get column names
rows = [list(dataset.columns)]       # column headers
rows += dataset.to_numpy().tolist()  # rows of data

# Write to spreadsheet
spreadsheet.append_rows(rows)

# Read the entire spreadsheet, except header
maxrow = spreadsheet.row_count
maxcol = chr(ord("A") - 1 + spreadsheet.col_count)
data = spreadsheet.get(f"A2:{maxcol}{maxrow}")
X = [row[:-1] for row in data]
y = [1 if row[-1]=="tested_positive" else 0 for row in data]
# 转换数据类型 / Convert data type
X, y = np.asarray(X).astype(float), np.asarray(y)

# create binary classification model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(16, input_dim=8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
# 训练模型 / Train the model
history = model.fit(X, y, epochs=5)
```

---

### Chapter Summary / 章节总结

# Chapter 25 Summary / 第25章总结

## Theme / 主题: Chapter 25 / Chapter 25

This chapter contains **11 code files** demonstrating chapter 25.

本章包含 **11 个代码文件**，演示Chapter 25。

---
## Evolution / 演化路线

  1. `08_sqlite3.ipynb` — Sqlite3
  2. `11_generator.ipynb` — Generator
  3. `15_dbm.ipynb` — Dbm
  4. `18_digits.ipynb` — Digits
  5. `19_excel.ipynb` — Excel
  6. `22_excel.ipynb` — Excel
  7. `23_excelkeras.ipynb` — Excelkeras
  8. `24_inmemory.ipynb` — Inmemory
  9. `31_googlesheet.ipynb` — Googlesheet
  10. `34_gspread.ipynb` — Gspread
  11. `35_gspreadkeras.ipynb` — Gspreadkeras

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 25) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 25）是机器学习流水线中的基础构建块。

---
