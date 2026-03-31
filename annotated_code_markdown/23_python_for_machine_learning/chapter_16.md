# Python 机器学习 / Python for Machine Learning
## Chapter 16

---

### Arg



---

### Argparse

# 03 — Argparse / 03 Argparse

**Chapter 16 — File 2 of 8 / 第16章 — 第2个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Argparse**.

本脚本演示 **03 Argparse**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import argparse

parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-a", "--archive", action="store_true", help="archive mode")
parser.add_argument("-v", "--verbose", action="store_true", help="increase verbosity")
parser.add_argument("-B", "--block-size", help="checksum blocksize")
parser.add_argument("--ignore-existing", action="store_true",
                    help="skip files that exist")
parser.add_argument("--exclude", help="files to exclude")
parser.add_argument("src", help="Source location")
parser.add_argument("dest", help="Destination location")
args = parser.parse_args()
config = vars(args)
# 打印输出 / Print output
print(config)
```

---
## Learning Notes / 学习笔记

- **概念**: Argparse 是机器学习中的常用技术。  
  *Argparse is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Argparse / 03 Argparse
# Complete Code / 完整代码
# ===============================

import argparse

parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-a", "--archive", action="store_true", help="archive mode")
parser.add_argument("-v", "--verbose", action="store_true", help="increase verbosity")
parser.add_argument("-B", "--block-size", help="checksum blocksize")
parser.add_argument("--ignore-existing", action="store_true",
                    help="skip files that exist")
parser.add_argument("--exclude", help="files to exclude")
parser.add_argument("src", help="Source location")
parser.add_argument("dest", help="Destination location")
args = parser.parse_args()
config = vars(args)
# 打印输出 / Print output
print(config)
```

---

➡️ **Next / 下一步**: File 3 of 8

---

### Gdp

# 07 — Gdp / 07 Gdp

**Chapter 16 — File 3 of 8 / 第16章 — 第3个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Gdp**.

本脚本演示 **07 Gdp**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas_datareader.wb import WorldBankReader

gdp = WorldBankReader("NY.GDP.MKTP.CN", "SE", start=1960, end=2020).read()
```

---
## Learning Notes / 学习笔记

- **概念**: Gdp 是机器学习中的常用技术。  
  *Gdp is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Gdp / 07 Gdp
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas_datareader.wb import WorldBankReader

gdp = WorldBankReader("NY.GDP.MKTP.CN", "SE", start=1960, end=2020).read()
```

---

➡️ **Next / 下一步**: File 4 of 8

---

### Arima

# 10 — Arima / ARIMA 模型

**Chapter 16 — File 4 of 8 / 第16章 — 第4个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Read the GDP data from WorldBank database**.

本脚本演示 **Read the GDP data from WorldBank database**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 训练模型 / Train the model


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏋️ 训练模型 / Train Model
```

---
## Step 1 — Step 1

```python
# 导入警告控制模块 / Import warnings control module
import warnings
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas_datareader.wb import WorldBankReader
import statsmodels.api as sm
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
warnings.simplefilter("ignore")

series = "NY.GDP.MKTP.CN"
country = "SE" # Sweden
length = 40
start = 0
steps = 3
order = (1,1,1)
```

---
## Step 2 — Read the GDP data from WorldBank database

```python
gdp = WorldBankReader(series, country, start=1960, end=2020).read()
```

---
## Step 3 — Drop country name from index

```python
gdp = gdp.droplevel(level=0, axis=0)
```

---
## Step 4 — Sort data in choronological order and set data point at year-end

```python
gdp.index = pd.to_datetime(gdp.index)
gdp = gdp.sort_index().resample("y").last()
```

---
## Step 5 — Convert pandas dataframe into pandas series

```python
gdp = gdp[series]
```

---
## Step 6 — Fit arima model

```python
result = sm.tsa.ARIMA(endog=gdp[start:start+length], order=order).fit()
```

---
## Step 7 — Forecast, and calculate the relative error

```python
forecast = result.forecast(steps=steps)
# 删除含缺失值的行 / Drop rows with missing values
df = pd.DataFrame({"Actual":gdp, "Forecast":forecast}).dropna()
df["Rel Error"] = (df["Forecast"] - df["Actual"]) / df["Actual"]
```

---
## Step 8 — Print result

```python
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    # 打印输出 / Print output
    print(df)
```

---
## Learning Notes / 学习笔记

- **概念**: Read the GDP data from WorldBank database 是机器学习中的常用技术。  
  *Read the GDP data from WorldBank database is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `dropna` | 删除缺失值 | Drop missing values |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Arima / ARIMA 模型
# Complete Code / 完整代码
# ===============================

# 导入警告控制模块 / Import warnings control module
import warnings
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas_datareader.wb import WorldBankReader
import statsmodels.api as sm
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
warnings.simplefilter("ignore")

series = "NY.GDP.MKTP.CN"
country = "SE" # Sweden
length = 40
start = 0
steps = 3
order = (1,1,1)

# Read the GDP data from WorldBank database
gdp = WorldBankReader(series, country, start=1960, end=2020).read()
# Drop country name from index
gdp = gdp.droplevel(level=0, axis=0)
# Sort data in choronological order and set data point at year-end
gdp.index = pd.to_datetime(gdp.index)
gdp = gdp.sort_index().resample("y").last()
# Convert pandas dataframe into pandas series
gdp = gdp[series]
# Fit arima model
result = sm.tsa.ARIMA(endog=gdp[start:start+length], order=order).fit()
# Forecast, and calculate the relative error
forecast = result.forecast(steps=steps)
# 删除含缺失值的行 / Drop rows with missing values
df = pd.DataFrame({"Actual":gdp, "Forecast":forecast}).dropna()
df["Rel Error"] = (df["Forecast"] - df["Actual"]) / df["Actual"]
# Print result
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    # 打印输出 / Print output
    print(df)
```

---

➡️ **Next / 下一步**: File 5 of 8

---

### Arima

# 11 — Arima / ARIMA 模型

**Chapter 16 — File 5 of 8 / 第16章 — 第5个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Parse command line arguments**.

本脚本演示 **Parse command line arguments**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 训练模型 / Train the model


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏋️ 训练模型 / Train Model
```

---
## Step 1 — Step 1

```python
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# 导入警告控制模块 / Import warnings control module
import warnings
warnings.simplefilter("ignore")

# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas_datareader.wb import WorldBankReader
import statsmodels.api as sm
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
```

---
## Step 2 — Parse command line arguments

```python
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--country", default="SE", help="Two-letter country code")
parser.add_argument("-l", "--length", default=40, type=int, help="Length of time series to fit the ARIMA model")
parser.add_argument("-s", "--start", default=0, type=int, help="Starting offset to fit the ARIMA model")
args = vars(parser.parse_args())
```

---
## Step 3 — Set up parameters

```python
series = "NY.GDP.MKTP.CN"
country = args["country"]
length = args["length"]
start = args["start"]
steps = 3
order = (1,1,1)
```

---
## Step 4 — Read the GDP data from WorldBank database

```python
gdp = WorldBankReader(series, country, start=1960, end=2020).read()
```

---
## Step 5 — Drop country name from index

```python
gdp = gdp.droplevel(level=0, axis=0)
```

---
## Step 6 — Sort data in choronological order and set data point at year-end

```python
gdp.index = pd.to_datetime(gdp.index)
gdp = gdp.sort_index().resample("y").last()
```

---
## Step 7 — Convert pandas dataframe into pandas series

```python
gdp = gdp[series]
```

---
## Step 8 — Fit arima model

```python
result = sm.tsa.ARIMA(endog=gdp[start:start+length], order=order).fit()
```

---
## Step 9 — Forecast, and calculate the relative error

```python
forecast = result.forecast(steps=steps)
# 删除含缺失值的行 / Drop rows with missing values
df = pd.DataFrame({"Actual":gdp, "Forecast":forecast}).dropna()
df["Rel Error"] = (df["Forecast"] - df["Actual"]) / df["Actual"]
```

---
## Step 10 — Print result

```python
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    # 打印输出 / Print output
    print(df)
```

---
## Learning Notes / 学习笔记

- **概念**: Parse command line arguments 是机器学习中的常用技术。  
  *Parse command line arguments is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `dropna` | 删除缺失值 | Drop missing values |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Arima / ARIMA 模型
# Complete Code / 完整代码
# ===============================

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# 导入警告控制模块 / Import warnings control module
import warnings
warnings.simplefilter("ignore")

# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas_datareader.wb import WorldBankReader
import statsmodels.api as sm
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--country", default="SE", help="Two-letter country code")
parser.add_argument("-l", "--length", default=40, type=int, help="Length of time series to fit the ARIMA model")
parser.add_argument("-s", "--start", default=0, type=int, help="Starting offset to fit the ARIMA model")
args = vars(parser.parse_args())

# Set up parameters
series = "NY.GDP.MKTP.CN"
country = args["country"]
length = args["length"]
start = args["start"]
steps = 3
order = (1,1,1)

# Read the GDP data from WorldBank database
gdp = WorldBankReader(series, country, start=1960, end=2020).read()
# Drop country name from index
gdp = gdp.droplevel(level=0, axis=0)
# Sort data in choronological order and set data point at year-end
gdp.index = pd.to_datetime(gdp.index)
gdp = gdp.sort_index().resample("y").last()
# Convert pandas dataframe into pandas series
gdp = gdp[series]
# Fit arima model
result = sm.tsa.ARIMA(endog=gdp[start:start+length], order=order).fit()
# Forecast, and calculate the relative error
forecast = result.forecast(steps=steps)
# 删除含缺失值的行 / Drop rows with missing values
df = pd.DataFrame({"Actual":gdp, "Forecast":forecast}).dropna()
df["Rel Error"] = (df["Forecast"] - df["Actual"]) / df["Actual"]
# Print result
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    # 打印输出 / Print output
    print(df)
```

---

➡️ **Next / 下一步**: File 6 of 8

---

### Env



---

### Yaml

# 21 — Yaml / 21 Yaml

**Chapter 16 — File 7 of 8 / 第16章 — 第7个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Load config from YAML file**.

本脚本演示 **Load config from YAML file**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 训练模型 / Train the model


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏋️ 训练模型 / Train Model
```

---
## Step 1 — Step 1

```python
# 导入警告控制模块 / Import warnings control module
import warnings
warnings.simplefilter("ignore")

# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas_datareader.wb import WorldBankReader
import statsmodels.api as sm
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import yaml
```

---
## Step 2 — Load config from YAML file

```python
# 打开文件（自动关闭） / Open file (auto-close)
with open("config.yaml", "r") as fp:
    args = yaml.safe_load(fp)
```

---
## Step 3 — Set up parameters

```python
series = "NY.GDP.MKTP.CN"
country = args["country"]
length = args["length"]
start = args["start"]
steps = 3
order = (1,1,1)
```

---
## Step 4 — Read the GDP data from WorldBank database

```python
gdp = WorldBankReader(series, country, start=1960, end=2020).read()
```

---
## Step 5 — Drop country name from index

```python
gdp = gdp.droplevel(level=0, axis=0)
```

---
## Step 6 — Sort data in choronological order and set data point at year-end

```python
gdp.index = pd.to_datetime(gdp.index)
gdp = gdp.sort_index().resample("y").last()
```

---
## Step 7 — Convert pandas dataframe into pandas series

```python
gdp = gdp[series]
```

---
## Step 8 — Fit arima model

```python
result = sm.tsa.ARIMA(endog=gdp[start:start+length], order=order).fit()
```

---
## Step 9 — Forecast, and calculate the relative error

```python
forecast = result.forecast(steps=steps)
# 删除含缺失值的行 / Drop rows with missing values
df = pd.DataFrame({"Actual":gdp, "Forecast":forecast}).dropna()
df["Rel Error"] = (df["Forecast"] - df["Actual"]) / df["Actual"]
```

---
## Step 10 — Print result

```python
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    # 打印输出 / Print output
    print(df)
```

---
## Learning Notes / 学习笔记

- **概念**: Load config from YAML file 是机器学习中的常用技术。  
  *Load config from YAML file is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `dropna` | 删除缺失值 | Drop missing values |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Yaml / 21 Yaml
# Complete Code / 完整代码
# ===============================

# 导入警告控制模块 / Import warnings control module
import warnings
warnings.simplefilter("ignore")

# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas_datareader.wb import WorldBankReader
import statsmodels.api as sm
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import yaml

# Load config from YAML file
# 打开文件（自动关闭） / Open file (auto-close)
with open("config.yaml", "r") as fp:
    args = yaml.safe_load(fp)

# Set up parameters
series = "NY.GDP.MKTP.CN"
country = args["country"]
length = args["length"]
start = args["start"]
steps = 3
order = (1,1,1)

# Read the GDP data from WorldBank database
gdp = WorldBankReader(series, country, start=1960, end=2020).read()
# Drop country name from index
gdp = gdp.droplevel(level=0, axis=0)
# Sort data in choronological order and set data point at year-end
gdp.index = pd.to_datetime(gdp.index)
gdp = gdp.sort_index().resample("y").last()
# Convert pandas dataframe into pandas series
gdp = gdp[series]
# Fit arima model
result = sm.tsa.ARIMA(endog=gdp[start:start+length], order=order).fit()
# Forecast, and calculate the relative error
forecast = result.forecast(steps=steps)
# 删除含缺失值的行 / Drop rows with missing values
df = pd.DataFrame({"Actual":gdp, "Forecast":forecast}).dropna()
df["Rel Error"] = (df["Forecast"] - df["Actual"]) / df["Actual"]
# Print result
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    # 打印输出 / Print output
    print(df)
```

---

➡️ **Next / 下一步**: File 8 of 8

---

### Json

# 23 — Json / 23 Json

**Chapter 16 — File 8 of 8 / 第16章 — 第8个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Load config from JSON file**.

本脚本演示 **Load config from JSON file**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 训练模型 / Train the model


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏋️ 训练模型 / Train Model
```

---
## Step 1 — Step 1

```python
# 导入JSON处理模块 / Import JSON processing module
import json
# 导入警告控制模块 / Import warnings control module
import warnings
warnings.simplefilter("ignore")

# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas_datareader.wb import WorldBankReader
import statsmodels.api as sm
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
```

---
## Step 2 — Load config from JSON file

```python
# 打开文件（自动关闭） / Open file (auto-close)
with open("config.json", "r") as fp:
    # 读取JSON文件 / Read JSON file
    args = json.load(fp)
```

---
## Step 3 — Set up parameters

```python
series = "NY.GDP.MKTP.CN"
country = args["country"]
length = args["length"]
start = args["start"]
steps = 3
order = (1,1,1)
```

---
## Step 4 — Read the GDP data from WorldBank database

```python
gdp = WorldBankReader(series, country, start=1960, end=2020).read()
```

---
## Step 5 — Drop country name from index

```python
gdp = gdp.droplevel(level=0, axis=0)
```

---
## Step 6 — Sort data in choronological order and set data point at year-end

```python
gdp.index = pd.to_datetime(gdp.index)
gdp = gdp.sort_index().resample("y").last()
```

---
## Step 7 — Convert pandas dataframe into pandas series

```python
gdp = gdp[series]
```

---
## Step 8 — Fit arima model

```python
result = sm.tsa.ARIMA(endog=gdp[start:start+length], order=order).fit()
```

---
## Step 9 — Forecast, and calculate the relative error

```python
forecast = result.forecast(steps=steps)
# 删除含缺失值的行 / Drop rows with missing values
df = pd.DataFrame({"Actual":gdp, "Forecast":forecast}).dropna()
df["Rel Error"] = (df["Forecast"] - df["Actual"]) / df["Actual"]
```

---
## Step 10 — Print result

```python
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    # 打印输出 / Print output
    print(df)
```

---
## Learning Notes / 学习笔记

- **概念**: Load config from JSON file 是机器学习中的常用技术。  
  *Load config from JSON file is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `dropna` | 删除缺失值 | Drop missing values |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Json / 23 Json
# Complete Code / 完整代码
# ===============================

# 导入JSON处理模块 / Import JSON processing module
import json
# 导入警告控制模块 / Import warnings control module
import warnings
warnings.simplefilter("ignore")

# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas_datareader.wb import WorldBankReader
import statsmodels.api as sm
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

# Load config from JSON file
# 打开文件（自动关闭） / Open file (auto-close)
with open("config.json", "r") as fp:
    # 读取JSON文件 / Read JSON file
    args = json.load(fp)

# Set up parameters
series = "NY.GDP.MKTP.CN"
country = args["country"]
length = args["length"]
start = args["start"]
steps = 3
order = (1,1,1)

# Read the GDP data from WorldBank database
gdp = WorldBankReader(series, country, start=1960, end=2020).read()
# Drop country name from index
gdp = gdp.droplevel(level=0, axis=0)
# Sort data in choronological order and set data point at year-end
gdp.index = pd.to_datetime(gdp.index)
gdp = gdp.sort_index().resample("y").last()
# Convert pandas dataframe into pandas series
gdp = gdp[series]
# Fit arima model
result = sm.tsa.ARIMA(endog=gdp[start:start+length], order=order).fit()
# Forecast, and calculate the relative error
forecast = result.forecast(steps=steps)
# 删除含缺失值的行 / Drop rows with missing values
df = pd.DataFrame({"Actual":gdp, "Forecast":forecast}).dropna()
df["Rel Error"] = (df["Forecast"] - df["Actual"]) / df["Actual"]
# Print result
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    # 打印输出 / Print output
    print(df)
```

---

### Chapter Summary / 章节总结



---
