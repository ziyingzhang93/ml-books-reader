# Python 机器学习 / Python for Machine Learning
## Chapter 28

---

### Readexcel

# 01 — Readexcel / 01 Readexcel

**Chapter 28 — File 1 of 20 / 第28章 — 第1个文件（共20个）**

---

## Summary / 总结

This script demonstrates **Readexcel**.

本脚本演示 **01 Readexcel**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

# 从Excel文件读取数据 / Read Excel file into DataFrame
df = pd.read_excel(URL, sheet_name="State_Trends", header=1)
# 打印输出 / Print output
print(df)
```

---
## Learning Notes / 学习笔记

- **概念**: Readexcel 是机器学习中的常用技术。  
  *Readexcel is a common technique in machine learning.*

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
# Readexcel / 01 Readexcel
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

# 从Excel文件读取数据 / Read Excel file into DataFrame
df = pd.read_excel(URL, sheet_name="State_Trends", header=1)
# 打印输出 / Print output
print(df)
```

---

➡️ **Next / 下一步**: File 2 of 20

---

### Info

# 02 — Info / 02 Info

**Chapter 28 — File 2 of 20 / 第28章 — 第2个文件（共20个）**

---

## Summary / 总结

This script demonstrates **Info**.

本脚本演示 **02 Info**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

# 从Excel文件读取数据 / Read Excel file into DataFrame
df = pd.read_excel(URL, sheet_name="State_Trends", header=1)
# 显示数据类型和缺失值信息 / Show data types and missing value info
df.info() # print info to screen
```

---
## Learning Notes / 学习笔记

- **概念**: Info 是机器学习中的常用技术。  
  *Info is a common technique in machine learning.*

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
# Info / 02 Info
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

# 从Excel文件读取数据 / Read Excel file into DataFrame
df = pd.read_excel(URL, sheet_name="State_Trends", header=1)
# 显示数据类型和缺失值信息 / Show data types and missing value info
df.info() # print info to screen
```

---

➡️ **Next / 下一步**: File 3 of 20

---

### Series

# 03 — Series / 03 Series

**Chapter 28 — File 3 of 20 / 第28章 — 第3个文件（共20个）**

---

## Summary / 总结

This script demonstrates **Series**.

本脚本演示 **03 Series**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

# 从Excel文件读取数据 / Read Excel file into DataFrame
df = pd.read_excel(URL, sheet_name="State_Trends", header=1)
coltypes = df.dtypes
# 打印输出 / Print output
print(coltypes)
```

---
## Learning Notes / 学习笔记

- **概念**: Series 是机器学习中的常用技术。  
  *Series is a common technique in machine learning.*

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
# Series / 03 Series
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

# 从Excel文件读取数据 / Read Excel file into DataFrame
df = pd.read_excel(URL, sheet_name="State_Trends", header=1)
coltypes = df.dtypes
# 打印输出 / Print output
print(coltypes)
```

---

➡️ **Next / 下一步**: File 4 of 20

---

### Fancy

# 04 — Fancy / 04 Fancy

**Chapter 28 — File 4 of 20 / 第28章 — 第4个文件（共20个）**

---

## Summary / 总结

This script demonstrates **Fancy**.

本脚本演示 **04 Fancy**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

# 从Excel文件读取数据 / Read Excel file into DataFrame
df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

cols = ["State", "Pollutant", "emissions19", "emissions20", "emissions21"]
last3years = df[cols]
# 打印输出 / Print output
print(last3years)
```

---
## Learning Notes / 学习笔记

- **概念**: Fancy 是机器学习中的常用技术。  
  *Fancy is a common technique in machine learning.*

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
# Fancy / 04 Fancy
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

# 从Excel文件读取数据 / Read Excel file into DataFrame
df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

cols = ["State", "Pollutant", "emissions19", "emissions20", "emissions21"]
last3years = df[cols]
# 打印输出 / Print output
print(last3years)
```

---

➡️ **Next / 下一步**: File 5 of 20

---

### Column

# 05 — Column / 05 Column

**Chapter 28 — File 5 of 20 / 第28章 — 第5个文件（共20个）**

---

## Summary / 总结

This script demonstrates **Column**.

本脚本演示 **05 Column**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

# 从Excel文件读取数据 / Read Excel file into DataFrame
df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

data2021 = df["emissions21"]
# 打印输出 / Print output
print(data2021)
```

---
## Learning Notes / 学习笔记

- **概念**: Column 是机器学习中的常用技术。  
  *Column is a common technique in machine learning.*

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
# Column / 05 Column
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

# 从Excel文件读取数据 / Read Excel file into DataFrame
df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

data2021 = df["emissions21"]
# 打印输出 / Print output
print(data2021)
```

---

➡️ **Next / 下一步**: File 6 of 20

---

### Unique

# 06 — Unique / 06 Unique

**Chapter 28 — File 6 of 20 / 第28章 — 第6个文件（共20个）**

---

## Summary / 总结

This script demonstrates **Unique**.

本脚本演示 **06 Unique**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

# 从Excel文件读取数据 / Read Excel file into DataFrame
df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

# 打印输出 / Print output
print(df["Pollutant"].unique())
```

---
## Learning Notes / 学习笔记

- **概念**: Unique 是机器学习中的常用技术。  
  *Unique is a common technique in machine learning.*

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
# Unique / 06 Unique
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

# 从Excel文件读取数据 / Read Excel file into DataFrame
df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

# 打印输出 / Print output
print(df["Pollutant"].unique())
```

---

➡️ **Next / 下一步**: File 7 of 20

---

### Mean

# 07 — Mean / 07 Mean

**Chapter 28 — File 7 of 20 / 第28章 — 第7个文件（共20个）**

---

## Summary / 总结

This script demonstrates **Mean**.

本脚本演示 **07 Mean**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

# 从Excel文件读取数据 / Read Excel file into DataFrame
df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

# 打印输出 / Print output
print(df["emissions21"].mean())
```

---
## Learning Notes / 学习笔记

- **概念**: Mean 是机器学习中的常用技术。  
  *Mean is a common technique in machine learning.*

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
# Mean / 07 Mean
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

# 从Excel文件读取数据 / Read Excel file into DataFrame
df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

# 打印输出 / Print output
print(df["emissions21"].mean())
```

---

➡️ **Next / 下一步**: File 8 of 20

---

### Describe



---

### Boolean



---

### Highway



---

### Iloc



---

### Pivot



---

### Melt



---

### Groupby



---

### Table



---

### Join

# 18 — Join / 18 Join

**Chapter 28 — File 16 of 20 / 第28章 — 第16个文件（共20个）**

---

## Summary / 总结

This script demonstrates **Join**.

本脚本演示 **18 Join**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

# 从Excel文件读取数据 / Read Excel file into DataFrame
df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_co = df[df["Pollutant"]=="CO"].groupby("State") \
                                 .sum()[["emissions21"]] \
                                 .rename(columns={"emissions21":"CO"})
df_so2 = df[df["Pollutant"]=="SO2"].groupby("State") \
                                   .sum()[["emissions21"]] \
                                   .rename(columns={"emissions21":"SO2"})
df_joined = df_co.join(df_so2)
# 打印输出 / Print output
print(df_joined)
```

---
## Learning Notes / 学习笔记

- **概念**: Join 是机器学习中的常用技术。  
  *Join is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `groupby` | 分组聚合 | Group and aggregate |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Join / 18 Join
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

# 从Excel文件读取数据 / Read Excel file into DataFrame
df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_co = df[df["Pollutant"]=="CO"].groupby("State") \
                                 .sum()[["emissions21"]] \
                                 .rename(columns={"emissions21":"CO"})
df_so2 = df[df["Pollutant"]=="SO2"].groupby("State") \
                                   .sum()[["emissions21"]] \
                                   .rename(columns={"emissions21":"SO2"})
df_joined = df_co.join(df_so2)
# 打印输出 / Print output
print(df_joined)
```

---

➡️ **Next / 下一步**: File 17 of 20

---

### Merge

# 19 — Merge / 19 Merge

**Chapter 28 — File 17 of 20 / 第28章 — 第17个文件（共20个）**

---

## Summary / 总结

This script demonstrates **Merge**.

本脚本演示 **19 Merge**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

# 从Excel文件读取数据 / Read Excel file into DataFrame
df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_co = df[df["Pollutant"]=="CO"].groupby("State") \
                                 .sum()[["emissions21"]] \
                                 .rename(columns={"emissions21":"CO"}) \
                                 .reset_index()
df_so2 = df[df["Pollutant"]=="SO2"].groupby("State") \
                                   .sum()[["emissions21"]] \
                                   .rename(columns={"emissions21":"SO2"}) \
                                   .reset_index()
df_merged = df_co.merge(df_so2, on="State", how="outer")
# 打印输出 / Print output
print(df_merged)
```

---
## Learning Notes / 学习笔记

- **概念**: Merge 是机器学习中的常用技术。  
  *Merge is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `groupby` | 分组聚合 | Group and aggregate |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Merge / 19 Merge
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

# 从Excel文件读取数据 / Read Excel file into DataFrame
df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_co = df[df["Pollutant"]=="CO"].groupby("State") \
                                 .sum()[["emissions21"]] \
                                 .rename(columns={"emissions21":"CO"}) \
                                 .reset_index()
df_so2 = df[df["Pollutant"]=="SO2"].groupby("State") \
                                   .sum()[["emissions21"]] \
                                   .rename(columns={"emissions21":"SO2"}) \
                                   .reset_index()
df_merged = df_co.merge(df_so2, on="State", how="outer")
# 打印输出 / Print output
print(df_merged)
```

---

➡️ **Next / 下一步**: File 18 of 20

---

### Apply



---

### Pandas

# 21 — Pandas / 21 Pandas

**Chapter 28 — File 19 of 20 / 第28章 — 第19个文件（共20个）**

---

## Summary / 总结

This script demonstrates **Pollutants data from Environmental Protection Agency**.

本脚本演示 **Pollutants data from Environmental Protection Agency**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
```

---
## Step 2 — Pollutants data from Environmental Protection Agency

```python
URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"
```

---
## Step 3 — Read the Excel file and print

```python
# 从Excel文件读取数据 / Read Excel file into DataFrame
df = pd.read_excel(URL, sheet_name="State_Trends", header=1)
# 打印输出 / Print output
print("US air pollutant emission data:")
# 打印输出 / Print output
print(df)
```

---
## Step 4 — Show info

```python
# 打印输出 / Print output
print("\nInformation about the DataFrame:")
# 显示数据类型和缺失值信息 / Show data types and missing value info
df.info()
```

---
## Step 5 — print dtyes

```python
coltypes = df.dtypes
# 打印输出 / Print output
print("\nColumn data types of the DataFrame:")
# 打印输出 / Print output
print(coltypes)
```

---
## Step 6 — Get last 3 columns

```python
cols = ["State", "Pollutant", "emissions19", "emissions20", "emissions21"]
last3years = df[cols]
# 打印输出 / Print output
print("\nDataFrame of last 3 years data:")
# 打印输出 / Print output
print(last3years)
```

---
## Step 7 — Get a series

```python
data2021 = df["emissions21"]
# 打印输出 / Print output
print("\nSeries of 2021 data:")
# 打印输出 / Print output
print(data2021)
```

---
## Step 8 — Print unique pollutants

```python
# 打印输出 / Print output
print("\nUnique pollutants:")
# 打印输出 / Print output
print(df["Pollutant"].unique())
```

---
## Step 9 — print mean emission

```python
# 打印输出 / Print output
print("\nMean on the 2021 series:")
# 打印输出 / Print output
print(df["emissions21"].mean())
```

---
## Step 10 — Describe

```python
# 打印输出 / Print output
print("\nBasic statistics about each column in the DataFrame:")
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(df.describe().T)
```

---
## Step 11 — Get CO only

```python
df_CO = df[df["Pollutant"] == "CO"]
# 打印输出 / Print output
print("\nDataFrame of only CO pollutant:")
# 打印输出 / Print output
print(df_CO)
```

---
## Step 12 — Get CO and Highway only

```python
df_CO_HW = df[(df["Pollutant"] == "CO")
              & (df["Tier 1 Description"] == "HIGHWAY VEHICLES")]
# 打印输出 / Print output
print("\nDataFrame of only CO pollutant from Highway vehicles:")
# 打印输出 / Print output
print(df_CO_HW)
```

---
## Step 13 — Get DF of all CO

```python
df_all_co = df[df["Pollutant"]=="CO"][["State", "Tier 1 Description", "emissions21"]]
# 打印输出 / Print output
print("\nDataFrame of only CO pollutant, keep only essential columns:")
# 打印输出 / Print output
print(df_all_co)
```

---
## Step 14 — Pivot

```python
df_pivot = df_all_co.pivot_table(index="State",
                                 columns="Tier 1 Description",
                                 values="emissions21")
# 打印输出 / Print output
print("\nPivot table of state vs CO emission source:")
# 打印输出 / Print output
print(df_pivot)
```

---
## Step 15 — melt

```python
df_melt = df_pivot.melt(value_name="emissions 2021",
                        var_name="Tier 1 Description",
                        ignore_index=False)
# 打印输出 / Print output
print("\nMelting the pivot table:")
# 打印输出 / Print output
print(df_melt)
```

---
## Step 16 — all three are the same

```python
# 填充缺失值 / Fill missing values
df_filled = df_pivot.fillna(0)
df_filled = df_pivot.where(df_pivot.notna(), 0)
df_filled = df_pivot.mask(df_pivot.isna(), 0)
# 打印输出 / Print output
print("\nFilled missing value as zero:")
# 打印输出 / Print output
print(df_filled)
```

---
## Step 17 — aggregation

```python
df_sum = df[df["Pollutant"]=="CO"].groupby("State").sum()
# 打印输出 / Print output
print("\nTotal CO emission by state:")
# 打印输出 / Print output
print(df_sum)
```

---
## Step 18 — group by

```python
df_2021 = ( df.groupby(["State", "Pollutant"])
              .sum()              # get total emissions of each year
              [["emissions21"]]   # select only year 2021
              .reset_index()
              .pivot(index="State", columns="Pollutant", values="emissions21")
              .filter(["CO","SO2"])
          )
# 打印输出 / Print output
print("\nComparing CO and SO2 emission:")
# 打印输出 / Print output
print(df_2021)
```

---
## Step 19 — join

```python
df_co = df[df["Pollutant"]=="CO"].groupby("State") \
                                 .sum()[["emissions21"]] \
                                 .rename(columns={"emissions21":"CO"})
df_so2 = df[df["Pollutant"]=="SO2"].groupby("State") \
                                   .sum()[["emissions21"]] \
                                   .rename(columns={"emissions21":"SO2"})
df_joined = df_co.join(df_so2)
# 打印输出 / Print output
print("\nComparing CO and SO2 emission:")
# 打印输出 / Print output
print(df_joined)
```

---
## Step 20 — merge

```python
df_co = df[df["Pollutant"]=="CO"].groupby("State") \
                                 .sum()[["emissions21"]] \
                                 .rename(columns={"emissions21":"CO"}) \
                                 .reset_index()
df_so2 = df[df["Pollutant"]=="SO2"].groupby("State") \
                                   .sum()[["emissions21"]] \
                                   .rename(columns={"emissions21":"SO2"}) \
                                   .reset_index()
df_merged = df_co.merge(df_so2, on="State", how="outer")
# 打印输出 / Print output
print("\nComparing CO and SO2 emission:")
# 打印输出 / Print output
print(df_merged)

def minmaxyear(subdf):
    sum_series = subdf.sum()
    year_indices = [x for x in sum_series.index if x.startswith("emissions")]
    # 转换数据类型 / Convert data type
    minyear = sum_series[year_indices].astype(float).idxmin()
    # 转换数据类型 / Convert data type
    maxyear = sum_series[year_indices].astype(float).idxmax()
    return pd.Series({"min year": minyear[-2:], "max year": maxyear[-2:]})

df_years = df[df["Pollutant"]=="CO"].groupby("State").apply(minmaxyear)
# 打印输出 / Print output
print("\nYears of minimum and maximum emissions:")
# 打印输出 / Print output
print(df_years)
```

---
## Learning Notes / 学习笔记

- **概念**: Pollutants data from Environmental Protection Agency 是机器学习中的常用技术。  
  *Pollutants data from Environmental Protection Agency is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `describe()` | 统计摘要信息 | Statistical summary |
| `fillna` | 填充缺失值 | Fill missing values |
| `groupby` | 分组聚合 | Group and aggregate |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pandas / 21 Pandas
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

# Pollutants data from Environmental Protection Agency
URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

# Read the Excel file and print
# 从Excel文件读取数据 / Read Excel file into DataFrame
df = pd.read_excel(URL, sheet_name="State_Trends", header=1)
# 打印输出 / Print output
print("US air pollutant emission data:")
# 打印输出 / Print output
print(df)

# Show info
# 打印输出 / Print output
print("\nInformation about the DataFrame:")
# 显示数据类型和缺失值信息 / Show data types and missing value info
df.info()

# print dtyes
coltypes = df.dtypes
# 打印输出 / Print output
print("\nColumn data types of the DataFrame:")
# 打印输出 / Print output
print(coltypes)

# Get last 3 columns
cols = ["State", "Pollutant", "emissions19", "emissions20", "emissions21"]
last3years = df[cols]
# 打印输出 / Print output
print("\nDataFrame of last 3 years data:")
# 打印输出 / Print output
print(last3years)

# Get a series
data2021 = df["emissions21"]
# 打印输出 / Print output
print("\nSeries of 2021 data:")
# 打印输出 / Print output
print(data2021)

# Print unique pollutants
# 打印输出 / Print output
print("\nUnique pollutants:")
# 打印输出 / Print output
print(df["Pollutant"].unique())

# print mean emission
# 打印输出 / Print output
print("\nMean on the 2021 series:")
# 打印输出 / Print output
print(df["emissions21"].mean())

# Describe
# 打印输出 / Print output
print("\nBasic statistics about each column in the DataFrame:")
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(df.describe().T)

# Get CO only
df_CO = df[df["Pollutant"] == "CO"]
# 打印输出 / Print output
print("\nDataFrame of only CO pollutant:")
# 打印输出 / Print output
print(df_CO)

# Get CO and Highway only
df_CO_HW = df[(df["Pollutant"] == "CO")
              & (df["Tier 1 Description"] == "HIGHWAY VEHICLES")]
# 打印输出 / Print output
print("\nDataFrame of only CO pollutant from Highway vehicles:")
# 打印输出 / Print output
print(df_CO_HW)

# Get DF of all CO
df_all_co = df[df["Pollutant"]=="CO"][["State", "Tier 1 Description", "emissions21"]]
# 打印输出 / Print output
print("\nDataFrame of only CO pollutant, keep only essential columns:")
# 打印输出 / Print output
print(df_all_co)

# Pivot
df_pivot = df_all_co.pivot_table(index="State",
                                 columns="Tier 1 Description",
                                 values="emissions21")
# 打印输出 / Print output
print("\nPivot table of state vs CO emission source:")
# 打印输出 / Print output
print(df_pivot)

# melt
df_melt = df_pivot.melt(value_name="emissions 2021",
                        var_name="Tier 1 Description",
                        ignore_index=False)
# 打印输出 / Print output
print("\nMelting the pivot table:")
# 打印输出 / Print output
print(df_melt)

# all three are the same
# 填充缺失值 / Fill missing values
df_filled = df_pivot.fillna(0)
df_filled = df_pivot.where(df_pivot.notna(), 0)
df_filled = df_pivot.mask(df_pivot.isna(), 0)
# 打印输出 / Print output
print("\nFilled missing value as zero:")
# 打印输出 / Print output
print(df_filled)

# aggregation
df_sum = df[df["Pollutant"]=="CO"].groupby("State").sum()
# 打印输出 / Print output
print("\nTotal CO emission by state:")
# 打印输出 / Print output
print(df_sum)

# group by
df_2021 = ( df.groupby(["State", "Pollutant"])
              .sum()              # get total emissions of each year
              [["emissions21"]]   # select only year 2021
              .reset_index()
              .pivot(index="State", columns="Pollutant", values="emissions21")
              .filter(["CO","SO2"])
          )
# 打印输出 / Print output
print("\nComparing CO and SO2 emission:")
# 打印输出 / Print output
print(df_2021)

# join
df_co = df[df["Pollutant"]=="CO"].groupby("State") \
                                 .sum()[["emissions21"]] \
                                 .rename(columns={"emissions21":"CO"})
df_so2 = df[df["Pollutant"]=="SO2"].groupby("State") \
                                   .sum()[["emissions21"]] \
                                   .rename(columns={"emissions21":"SO2"})
df_joined = df_co.join(df_so2)
# 打印输出 / Print output
print("\nComparing CO and SO2 emission:")
# 打印输出 / Print output
print(df_joined)

# merge
df_co = df[df["Pollutant"]=="CO"].groupby("State") \
                                 .sum()[["emissions21"]] \
                                 .rename(columns={"emissions21":"CO"}) \
                                 .reset_index()
df_so2 = df[df["Pollutant"]=="SO2"].groupby("State") \
                                   .sum()[["emissions21"]] \
                                   .rename(columns={"emissions21":"SO2"}) \
                                   .reset_index()
df_merged = df_co.merge(df_so2, on="State", how="outer")
# 打印输出 / Print output
print("\nComparing CO and SO2 emission:")
# 打印输出 / Print output
print(df_merged)

def minmaxyear(subdf):
    sum_series = subdf.sum()
    year_indices = [x for x in sum_series.index if x.startswith("emissions")]
    # 转换数据类型 / Convert data type
    minyear = sum_series[year_indices].astype(float).idxmin()
    # 转换数据类型 / Convert data type
    maxyear = sum_series[year_indices].astype(float).idxmax()
    return pd.Series({"min year": minyear[-2:], "max year": maxyear[-2:]})

df_years = df[df["Pollutant"]=="CO"].groupby("State").apply(minmaxyear)
# 打印输出 / Print output
print("\nYears of minimum and maximum emissions:")
# 打印输出 / Print output
print(df_years)
```

---

➡️ **Next / 下一步**: File 20 of 20

---

### Timeseries

# 33 — Timeseries / 33 Timeseries

**Chapter 28 — File 20 of 20 / 第28章 — 第20个文件（共20个）**

---

## Summary / 总结

This script demonstrates **Load time series**.

本脚本演示 **Load time series**。

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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
```

---
## Step 2 — Load time series

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
df = pd.read_csv("ad_viz_plotval_data.csv", parse_dates=[0])
# 打印输出 / Print output
print("Input data:")
# 打印输出 / Print output
print(df)
```

---
## Step 3 — Set date index

```python
df_pm25 = df.set_index("Date")
# 打印输出 / Print output
print("\nUsing date index:")
# 打印输出 / Print output
print(df_pm25)
# 打印输出 / Print output
print(df_pm25.index)
```

---
## Step 4 — 2021 daily

```python
df_2021 = ( df[["Date", "Daily Mean PM2.5 Concentration", "Site Name"]]
            .pivot_table(index="Date",
                         columns="Site Name",
                         values="Daily Mean PM2.5 Concentration")
          )
# 打印输出 / Print output
print("\nUsing date index:")
# 打印输出 / Print output
print(df_2021)
# 打印输出 / Print output
print(df_2021.index.is_unique)
```

---
## Step 5 — Time interval

```python
df_3mon = df_2021["2021-04-01":"2021-07-01"]
# 打印输出 / Print output
print("\nInterval selection:")
# 打印输出 / Print output
print(df_3mon)
```

---
## Step 6 — Resample

```python
# 打印输出 / Print output
print("\nResampling dataframe:")
df_resample = df_2021.resample("W-SUN").first()
# 打印输出 / Print output
print(df_resample)
# 打印输出 / Print output
print("\nResampling series for OHLC:")
df_ohlc = df_2021["San Antonio Interstate 35"].resample("W-SUN").ohlc()
# 打印输出 / Print output
print(df_ohlc)
# 打印输出 / Print output
print("\nResampling series with forward fill:")
series_ffill = df_2021["San Antonio Interstate 35"].resample("H").ffill()
# 打印输出 / Print output
print(series_ffill)
```

---
## Step 7 — rolling

```python
# 打印输出 / Print output
print("\nRolling mean:")
df_mean = df_2021["San Antonio Interstate 35"].rolling(10).mean()
# 打印输出 / Print output
print(df_mean)
```

---
## Step 8 — Plot moving average

```python
# 创建画布 / Create figure canvas
fig = plt.figure(figsize=(12,6))
# 绘制折线图 / Draw line plot
plt.plot(df_2021["San Antonio Interstate 35"], label="daily")
# 绘制折线图 / Draw line plot
plt.plot(df_2021["San Antonio Interstate 35"].rolling(10, min_periods=5).mean(),
         label="10-day MA")
# 显示图例 / Show legend
plt.legend()
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("PM 2.5")
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Load time series 是机器学习中的常用技术。  
  *Load time series is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Timeseries / 33 Timeseries
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# Load time series
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
df = pd.read_csv("ad_viz_plotval_data.csv", parse_dates=[0])
# 打印输出 / Print output
print("Input data:")
# 打印输出 / Print output
print(df)

# Set date index
df_pm25 = df.set_index("Date")
# 打印输出 / Print output
print("\nUsing date index:")
# 打印输出 / Print output
print(df_pm25)
# 打印输出 / Print output
print(df_pm25.index)

# 2021 daily
df_2021 = ( df[["Date", "Daily Mean PM2.5 Concentration", "Site Name"]]
            .pivot_table(index="Date",
                         columns="Site Name",
                         values="Daily Mean PM2.5 Concentration")
          )
# 打印输出 / Print output
print("\nUsing date index:")
# 打印输出 / Print output
print(df_2021)
# 打印输出 / Print output
print(df_2021.index.is_unique)

# Time interval
df_3mon = df_2021["2021-04-01":"2021-07-01"]
# 打印输出 / Print output
print("\nInterval selection:")
# 打印输出 / Print output
print(df_3mon)

# Resample
# 打印输出 / Print output
print("\nResampling dataframe:")
df_resample = df_2021.resample("W-SUN").first()
# 打印输出 / Print output
print(df_resample)
# 打印输出 / Print output
print("\nResampling series for OHLC:")
df_ohlc = df_2021["San Antonio Interstate 35"].resample("W-SUN").ohlc()
# 打印输出 / Print output
print(df_ohlc)
# 打印输出 / Print output
print("\nResampling series with forward fill:")
series_ffill = df_2021["San Antonio Interstate 35"].resample("H").ffill()
# 打印输出 / Print output
print(series_ffill)

# rolling
# 打印输出 / Print output
print("\nRolling mean:")
df_mean = df_2021["San Antonio Interstate 35"].rolling(10).mean()
# 打印输出 / Print output
print(df_mean)

# Plot moving average
# 创建画布 / Create figure canvas
fig = plt.figure(figsize=(12,6))
# 绘制折线图 / Draw line plot
plt.plot(df_2021["San Antonio Interstate 35"], label="daily")
# 绘制折线图 / Draw line plot
plt.plot(df_2021["San Antonio Interstate 35"].rolling(10, min_periods=5).mean(),
         label="10-day MA")
# 显示图例 / Show legend
plt.legend()
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("PM 2.5")
# 显示图表 / Display the plot
plt.show()
```

---

### Chapter Summary / 章节总结



---
