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
## Step 1 — Step 1

```python
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)
print(df)
```

---
## Learning Notes / 学习笔记

- **概念**: Readexcel 是机器学习中的常用技术。  
  *Readexcel is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Readexcel / 01 Readexcel
# Complete Code / 完整代码
# ===============================

import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)
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
## Step 1 — Step 1

```python
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)
df.info() # print info to screen
```

---
## Learning Notes / 学习笔记

- **概念**: Info 是机器学习中的常用技术。  
  *Info is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Info / 02 Info
# Complete Code / 完整代码
# ===============================

import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)
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
## Step 1 — Step 1

```python
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)
coltypes = df.dtypes
print(coltypes)
```

---
## Learning Notes / 学习笔记

- **概念**: Series 是机器学习中的常用技术。  
  *Series is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Series / 03 Series
# Complete Code / 完整代码
# ===============================

import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)
coltypes = df.dtypes
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
## Step 1 — Step 1

```python
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

cols = ["State", "Pollutant", "emissions19", "emissions20", "emissions21"]
last3years = df[cols]
print(last3years)
```

---
## Learning Notes / 学习笔记

- **概念**: Fancy 是机器学习中的常用技术。  
  *Fancy is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Fancy / 04 Fancy
# Complete Code / 完整代码
# ===============================

import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

cols = ["State", "Pollutant", "emissions19", "emissions20", "emissions21"]
last3years = df[cols]
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
## Step 1 — Step 1

```python
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

data2021 = df["emissions21"]
print(data2021)
```

---
## Learning Notes / 学习笔记

- **概念**: Column 是机器学习中的常用技术。  
  *Column is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Column / 05 Column
# Complete Code / 完整代码
# ===============================

import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

data2021 = df["emissions21"]
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
## Step 1 — Step 1

```python
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

print(df["Pollutant"].unique())
```

---
## Learning Notes / 学习笔记

- **概念**: Unique 是机器学习中的常用技术。  
  *Unique is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Unique / 06 Unique
# Complete Code / 完整代码
# ===============================

import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

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
## Step 1 — Step 1

```python
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

print(df["emissions21"].mean())
```

---
## Learning Notes / 学习笔记

- **概念**: Mean 是机器学习中的常用技术。  
  *Mean is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mean / 07 Mean
# Complete Code / 完整代码
# ===============================

import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

print(df["emissions21"].mean())
```

---

➡️ **Next / 下一步**: File 8 of 20

---

### Describe

# 08 — Describe / 08 Describe

**Chapter 28 — File 8 of 20 / 第28章 — 第8个文件（共20个）**

---

## Summary / 总结

This script demonstrates **Describe**.

本脚本演示 **08 Describe**。

---
## Step 1 — Step 1

```python
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

print(df.describe().T)
```

---
## Learning Notes / 学习笔记

- **概念**: Describe 是机器学习中的常用技术。  
  *Describe is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Describe / 08 Describe
# Complete Code / 完整代码
# ===============================

import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

print(df.describe().T)
```

---

➡️ **Next / 下一步**: File 9 of 20

---

### Boolean

# 09 — Boolean / 09 Boolean

**Chapter 28 — File 9 of 20 / 第28章 — 第9个文件（共20个）**

---

## Summary / 总结

This script demonstrates **Boolean**.

本脚本演示 **09 Boolean**。

---
## Step 1 — Step 1

```python
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_CO = df[df["Pollutant"] == "CO"]
print(df_CO)
```

---
## Learning Notes / 学习笔记

- **概念**: Boolean 是机器学习中的常用技术。  
  *Boolean is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Boolean / 09 Boolean
# Complete Code / 完整代码
# ===============================

import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_CO = df[df["Pollutant"] == "CO"]
print(df_CO)
```

---

➡️ **Next / 下一步**: File 10 of 20

---

### Highway

# 10 — Highway / 10 Highway

**Chapter 28 — File 10 of 20 / 第28章 — 第10个文件（共20个）**

---

## Summary / 总结

This script demonstrates **Highway**.

本脚本演示 **10 Highway**。

---
## Step 1 — Step 1

```python
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_CO_HW = df[(df["Pollutant"] == "CO") & (df["Tier 1 Description"] == "HIGHWAY VEHICLES")]
print(df_CO_HW)
```

---
## Learning Notes / 学习笔记

- **概念**: Highway 是机器学习中的常用技术。  
  *Highway is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Highway / 10 Highway
# Complete Code / 完整代码
# ===============================

import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_CO_HW = df[(df["Pollutant"] == "CO") & (df["Tier 1 Description"] == "HIGHWAY VEHICLES")]
print(df_CO_HW)
```

---

➡️ **Next / 下一步**: File 11 of 20

---

### Iloc

# 11 — Iloc / 11 Iloc

**Chapter 28 — File 11 of 20 / 第28章 — 第11个文件（共20个）**

---

## Summary / 总结

This script demonstrates **Iloc**.

本脚本演示 **11 Iloc**。

---
## Step 1 — Step 1

```python
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_r5 = df.iloc[5:11]
df_c1_r5 = df.iloc[5:11, 1:7]
```

---
## Learning Notes / 学习笔记

- **概念**: Iloc 是机器学习中的常用技术。  
  *Iloc is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Iloc / 11 Iloc
# Complete Code / 完整代码
# ===============================

import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_r5 = df.iloc[5:11]
df_c1_r5 = df.iloc[5:11, 1:7]
```

---

➡️ **Next / 下一步**: File 12 of 20

---

### Pivot

# 13 — Pivot / 13 Pivot

**Chapter 28 — File 12 of 20 / 第28章 — 第12个文件（共20个）**

---

## Summary / 总结

This script demonstrates **Pivot**.

本脚本演示 **13 Pivot**。

---
## Step 1 — Step 1

```python
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_all_co = df[df["Pollutant"]=="CO"][["State", "Tier 1 Description", "emissions21"]]
print(df_all_co)

df_pivot = df_all_co.pivot_table(index="State", columns="Tier 1 Description", values="emissions21")
print(df_pivot)
```

---
## Learning Notes / 学习笔记

- **概念**: Pivot 是机器学习中的常用技术。  
  *Pivot is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pivot / 13 Pivot
# Complete Code / 完整代码
# ===============================

import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_all_co = df[df["Pollutant"]=="CO"][["State", "Tier 1 Description", "emissions21"]]
print(df_all_co)

df_pivot = df_all_co.pivot_table(index="State", columns="Tier 1 Description", values="emissions21")
print(df_pivot)
```

---

➡️ **Next / 下一步**: File 13 of 20

---

### Melt

# 14 — Melt / 14 Melt

**Chapter 28 — File 13 of 20 / 第28章 — 第13个文件（共20个）**

---

## Summary / 总结

This script demonstrates **Melt**.

本脚本演示 **14 Melt**。

---
## Step 1 — Step 1

```python
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_all_co = df[df["Pollutant"]=="CO"][["State", "Tier 1 Description", "emissions21"]]
df_pivot = df_all_co.pivot_table(index="State", columns="Tier 1 Description", values="emissions21")
df_melt = df_pivot.melt(value_name="emissions 2021", var_name="Tier 1 Description", ignore_index=False)
print(df_melt)
```

---
## Learning Notes / 学习笔记

- **概念**: Melt 是机器学习中的常用技术。  
  *Melt is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Melt / 14 Melt
# Complete Code / 完整代码
# ===============================

import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_all_co = df[df["Pollutant"]=="CO"][["State", "Tier 1 Description", "emissions21"]]
df_pivot = df_all_co.pivot_table(index="State", columns="Tier 1 Description", values="emissions21")
df_melt = df_pivot.melt(value_name="emissions 2021", var_name="Tier 1 Description", ignore_index=False)
print(df_melt)
```

---

➡️ **Next / 下一步**: File 14 of 20

---

### Groupby

# 16 — Groupby / 16 Groupby

**Chapter 28 — File 14 of 20 / 第28章 — 第14个文件（共20个）**

---

## Summary / 总结

This script demonstrates **Groupby**.

本脚本演示 **16 Groupby**。

---
## Step 1 — Step 1

```python
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_sum = df[df["Pollutant"]=="CO"].groupby("State").sum()
print(df_sum)
```

---
## Learning Notes / 学习笔记

- **概念**: Groupby 是机器学习中的常用技术。  
  *Groupby is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Groupby / 16 Groupby
# Complete Code / 完整代码
# ===============================

import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_sum = df[df["Pollutant"]=="CO"].groupby("State").sum()
print(df_sum)
```

---

➡️ **Next / 下一步**: File 15 of 20

---

### Table

# 17 — Table / 17 Table

**Chapter 28 — File 15 of 20 / 第28章 — 第15个文件（共20个）**

---

## Summary / 总结

This script demonstrates **Table**.

本脚本演示 **17 Table**。

---
## Step 1 — Step 1

```python
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_2021 = ( df.groupby(["State", "Pollutant"])
              .sum()              # get total emissions of each year
              [["emissions21"]]   # select only year 2021
              .reset_index()
              .pivot(index="State", columns="Pollutant", values="emissions21")
              .filter(["CO","SO2"])
          )
print(df_2021)
```

---
## Learning Notes / 学习笔记

- **概念**: Table 是机器学习中的常用技术。  
  *Table is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Table / 17 Table
# Complete Code / 完整代码
# ===============================

import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_2021 = ( df.groupby(["State", "Pollutant"])
              .sum()              # get total emissions of each year
              [["emissions21"]]   # select only year 2021
              .reset_index()
              .pivot(index="State", columns="Pollutant", values="emissions21")
              .filter(["CO","SO2"])
          )
print(df_2021)
```

---

➡️ **Next / 下一步**: File 16 of 20

---

### Join

# 18 — Join / 18 Join

**Chapter 28 — File 16 of 20 / 第28章 — 第16个文件（共20个）**

---

## Summary / 总结

This script demonstrates **Join**.

本脚本演示 **18 Join**。

---
## Step 1 — Step 1

```python
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_co = df[df["Pollutant"]=="CO"].groupby("State") \
                                 .sum()[["emissions21"]] \
                                 .rename(columns={"emissions21":"CO"})
df_so2 = df[df["Pollutant"]=="SO2"].groupby("State") \
                                   .sum()[["emissions21"]] \
                                   .rename(columns={"emissions21":"SO2"})
df_joined = df_co.join(df_so2)
print(df_joined)
```

---
## Learning Notes / 学习笔记

- **概念**: Join 是机器学习中的常用技术。  
  *Join is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Join / 18 Join
# Complete Code / 完整代码
# ===============================

import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

df_co = df[df["Pollutant"]=="CO"].groupby("State") \
                                 .sum()[["emissions21"]] \
                                 .rename(columns={"emissions21":"CO"})
df_so2 = df[df["Pollutant"]=="SO2"].groupby("State") \
                                   .sum()[["emissions21"]] \
                                   .rename(columns={"emissions21":"SO2"})
df_joined = df_co.join(df_so2)
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
## Step 1 — Step 1

```python
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

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
print(df_merged)
```

---
## Learning Notes / 学习笔记

- **概念**: Merge 是机器学习中的常用技术。  
  *Merge is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Merge / 19 Merge
# Complete Code / 完整代码
# ===============================

import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

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
print(df_merged)
```

---

➡️ **Next / 下一步**: File 18 of 20

---

### Apply

# 20 — Apply / 20 Apply

**Chapter 28 — File 18 of 20 / 第28章 — 第18个文件（共20个）**

---

## Summary / 总结

This script demonstrates **Apply**.

本脚本演示 **20 Apply**。

---
## Step 1 — Step 1

```python
import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

def minmaxyear(subdf):
    sum_series = subdf.sum()
    year_indices = [x for x in sum_series.index if x.startswith("emissions")]
    minyear = sum_series[year_indices].astype(float).idxmin()
    maxyear = sum_series[year_indices].astype(float).idxmax()
    return pd.Series({"min year": minyear[-2:], "max year": maxyear[-2:]})

df_years = df[df["Pollutant"]=="CO"].groupby("State").apply(minmaxyear)
print(df_years)
```

---
## Learning Notes / 学习笔记

- **概念**: Apply 是机器学习中的常用技术。  
  *Apply is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Apply / 20 Apply
# Complete Code / 完整代码
# ===============================

import pandas as pd

URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

df = pd.read_excel(URL, sheet_name="State_Trends", header=1)

def minmaxyear(subdf):
    sum_series = subdf.sum()
    year_indices = [x for x in sum_series.index if x.startswith("emissions")]
    minyear = sum_series[year_indices].astype(float).idxmin()
    maxyear = sum_series[year_indices].astype(float).idxmax()
    return pd.Series({"min year": minyear[-2:], "max year": maxyear[-2:]})

df_years = df[df["Pollutant"]=="CO"].groupby("State").apply(minmaxyear)
print(df_years)
```

---

➡️ **Next / 下一步**: File 19 of 20

---

### Pandas

# 21 — Pandas / 21 Pandas

**Chapter 28 — File 19 of 20 / 第28章 — 第19个文件（共20个）**

---

## Summary / 总结

This script demonstrates **Pollutants data from Environmental Protection Agency**.

本脚本演示 **Pollutants data from Environmental Protection Agency**。

---
## Step 1 — Step 1

```python
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
df = pd.read_excel(URL, sheet_name="State_Trends", header=1)
print("US air pollutant emission data:")
print(df)
```

---
## Step 4 — Show info

```python
print("\nInformation about the DataFrame:")
df.info()
```

---
## Step 5 — print dtyes

```python
coltypes = df.dtypes
print("\nColumn data types of the DataFrame:")
print(coltypes)
```

---
## Step 6 — Get last 3 columns

```python
cols = ["State", "Pollutant", "emissions19", "emissions20", "emissions21"]
last3years = df[cols]
print("\nDataFrame of last 3 years data:")
print(last3years)
```

---
## Step 7 — Get a series

```python
data2021 = df["emissions21"]
print("\nSeries of 2021 data:")
print(data2021)
```

---
## Step 8 — Print unique pollutants

```python
print("\nUnique pollutants:")
print(df["Pollutant"].unique())
```

---
## Step 9 — print mean emission

```python
print("\nMean on the 2021 series:")
print(df["emissions21"].mean())
```

---
## Step 10 — Describe

```python
print("\nBasic statistics about each column in the DataFrame:")
print(df.describe().T)
```

---
## Step 11 — Get CO only

```python
df_CO = df[df["Pollutant"] == "CO"]
print("\nDataFrame of only CO pollutant:")
print(df_CO)
```

---
## Step 12 — Get CO and Highway only

```python
df_CO_HW = df[(df["Pollutant"] == "CO")
              & (df["Tier 1 Description"] == "HIGHWAY VEHICLES")]
print("\nDataFrame of only CO pollutant from Highway vehicles:")
print(df_CO_HW)
```

---
## Step 13 — Get DF of all CO

```python
df_all_co = df[df["Pollutant"]=="CO"][["State", "Tier 1 Description", "emissions21"]]
print("\nDataFrame of only CO pollutant, keep only essential columns:")
print(df_all_co)
```

---
## Step 14 — Pivot

```python
df_pivot = df_all_co.pivot_table(index="State",
                                 columns="Tier 1 Description",
                                 values="emissions21")
print("\nPivot table of state vs CO emission source:")
print(df_pivot)
```

---
## Step 15 — melt

```python
df_melt = df_pivot.melt(value_name="emissions 2021",
                        var_name="Tier 1 Description",
                        ignore_index=False)
print("\nMelting the pivot table:")
print(df_melt)
```

---
## Step 16 — all three are the same

```python
df_filled = df_pivot.fillna(0)
df_filled = df_pivot.where(df_pivot.notna(), 0)
df_filled = df_pivot.mask(df_pivot.isna(), 0)
print("\nFilled missing value as zero:")
print(df_filled)
```

---
## Step 17 — aggregation

```python
df_sum = df[df["Pollutant"]=="CO"].groupby("State").sum()
print("\nTotal CO emission by state:")
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
print("\nComparing CO and SO2 emission:")
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
print("\nComparing CO and SO2 emission:")
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
print("\nComparing CO and SO2 emission:")
print(df_merged)

def minmaxyear(subdf):
    sum_series = subdf.sum()
    year_indices = [x for x in sum_series.index if x.startswith("emissions")]
    minyear = sum_series[year_indices].astype(float).idxmin()
    maxyear = sum_series[year_indices].astype(float).idxmax()
    return pd.Series({"min year": minyear[-2:], "max year": maxyear[-2:]})

df_years = df[df["Pollutant"]=="CO"].groupby("State").apply(minmaxyear)
print("\nYears of minimum and maximum emissions:")
print(df_years)
```

---
## Learning Notes / 学习笔记

- **概念**: Pollutants data from Environmental Protection Agency 是机器学习中的常用技术。  
  *Pollutants data from Environmental Protection Agency is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pandas / 21 Pandas
# Complete Code / 完整代码
# ===============================

import pandas as pd

# Pollutants data from Environmental Protection Agency
URL = "https://www.epa.gov/sites/default/files/2021-03/state_tier1_caps.xlsx"

# Read the Excel file and print
df = pd.read_excel(URL, sheet_name="State_Trends", header=1)
print("US air pollutant emission data:")
print(df)

# Show info
print("\nInformation about the DataFrame:")
df.info()

# print dtyes
coltypes = df.dtypes
print("\nColumn data types of the DataFrame:")
print(coltypes)

# Get last 3 columns
cols = ["State", "Pollutant", "emissions19", "emissions20", "emissions21"]
last3years = df[cols]
print("\nDataFrame of last 3 years data:")
print(last3years)

# Get a series
data2021 = df["emissions21"]
print("\nSeries of 2021 data:")
print(data2021)

# Print unique pollutants
print("\nUnique pollutants:")
print(df["Pollutant"].unique())

# print mean emission
print("\nMean on the 2021 series:")
print(df["emissions21"].mean())

# Describe
print("\nBasic statistics about each column in the DataFrame:")
print(df.describe().T)

# Get CO only
df_CO = df[df["Pollutant"] == "CO"]
print("\nDataFrame of only CO pollutant:")
print(df_CO)

# Get CO and Highway only
df_CO_HW = df[(df["Pollutant"] == "CO")
              & (df["Tier 1 Description"] == "HIGHWAY VEHICLES")]
print("\nDataFrame of only CO pollutant from Highway vehicles:")
print(df_CO_HW)

# Get DF of all CO
df_all_co = df[df["Pollutant"]=="CO"][["State", "Tier 1 Description", "emissions21"]]
print("\nDataFrame of only CO pollutant, keep only essential columns:")
print(df_all_co)

# Pivot
df_pivot = df_all_co.pivot_table(index="State",
                                 columns="Tier 1 Description",
                                 values="emissions21")
print("\nPivot table of state vs CO emission source:")
print(df_pivot)

# melt
df_melt = df_pivot.melt(value_name="emissions 2021",
                        var_name="Tier 1 Description",
                        ignore_index=False)
print("\nMelting the pivot table:")
print(df_melt)

# all three are the same
df_filled = df_pivot.fillna(0)
df_filled = df_pivot.where(df_pivot.notna(), 0)
df_filled = df_pivot.mask(df_pivot.isna(), 0)
print("\nFilled missing value as zero:")
print(df_filled)

# aggregation
df_sum = df[df["Pollutant"]=="CO"].groupby("State").sum()
print("\nTotal CO emission by state:")
print(df_sum)

# group by
df_2021 = ( df.groupby(["State", "Pollutant"])
              .sum()              # get total emissions of each year
              [["emissions21"]]   # select only year 2021
              .reset_index()
              .pivot(index="State", columns="Pollutant", values="emissions21")
              .filter(["CO","SO2"])
          )
print("\nComparing CO and SO2 emission:")
print(df_2021)

# join
df_co = df[df["Pollutant"]=="CO"].groupby("State") \
                                 .sum()[["emissions21"]] \
                                 .rename(columns={"emissions21":"CO"})
df_so2 = df[df["Pollutant"]=="SO2"].groupby("State") \
                                   .sum()[["emissions21"]] \
                                   .rename(columns={"emissions21":"SO2"})
df_joined = df_co.join(df_so2)
print("\nComparing CO and SO2 emission:")
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
print("\nComparing CO and SO2 emission:")
print(df_merged)

def minmaxyear(subdf):
    sum_series = subdf.sum()
    year_indices = [x for x in sum_series.index if x.startswith("emissions")]
    minyear = sum_series[year_indices].astype(float).idxmin()
    maxyear = sum_series[year_indices].astype(float).idxmax()
    return pd.Series({"min year": minyear[-2:], "max year": maxyear[-2:]})

df_years = df[df["Pollutant"]=="CO"].groupby("State").apply(minmaxyear)
print("\nYears of minimum and maximum emissions:")
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
## Step 1 — Step 1

```python
import pandas as pd
import matplotlib.pyplot as plt
```

---
## Step 2 — Load time series

```python
df = pd.read_csv("ad_viz_plotval_data.csv", parse_dates=[0])
print("Input data:")
print(df)
```

---
## Step 3 — Set date index

```python
df_pm25 = df.set_index("Date")
print("\nUsing date index:")
print(df_pm25)
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
print("\nUsing date index:")
print(df_2021)
print(df_2021.index.is_unique)
```

---
## Step 5 — Time interval

```python
df_3mon = df_2021["2021-04-01":"2021-07-01"]
print("\nInterval selection:")
print(df_3mon)
```

---
## Step 6 — Resample

```python
print("\nResampling dataframe:")
df_resample = df_2021.resample("W-SUN").first()
print(df_resample)
print("\nResampling series for OHLC:")
df_ohlc = df_2021["San Antonio Interstate 35"].resample("W-SUN").ohlc()
print(df_ohlc)
print("\nResampling series with forward fill:")
series_ffill = df_2021["San Antonio Interstate 35"].resample("H").ffill()
print(series_ffill)
```

---
## Step 7 — rolling

```python
print("\nRolling mean:")
df_mean = df_2021["San Antonio Interstate 35"].rolling(10).mean()
print(df_mean)
```

---
## Step 8 — Plot moving average

```python
fig = plt.figure(figsize=(12,6))
plt.plot(df_2021["San Antonio Interstate 35"], label="daily")
plt.plot(df_2021["San Antonio Interstate 35"].rolling(10, min_periods=5).mean(),
         label="10-day MA")
plt.legend()
plt.ylabel("PM 2.5")
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Load time series 是机器学习中的常用技术。  
  *Load time series is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Timeseries / 33 Timeseries
# Complete Code / 完整代码
# ===============================

import pandas as pd
import matplotlib.pyplot as plt

# Load time series
df = pd.read_csv("ad_viz_plotval_data.csv", parse_dates=[0])
print("Input data:")
print(df)

# Set date index
df_pm25 = df.set_index("Date")
print("\nUsing date index:")
print(df_pm25)
print(df_pm25.index)

# 2021 daily
df_2021 = ( df[["Date", "Daily Mean PM2.5 Concentration", "Site Name"]]
            .pivot_table(index="Date",
                         columns="Site Name",
                         values="Daily Mean PM2.5 Concentration")
          )
print("\nUsing date index:")
print(df_2021)
print(df_2021.index.is_unique)

# Time interval
df_3mon = df_2021["2021-04-01":"2021-07-01"]
print("\nInterval selection:")
print(df_3mon)

# Resample
print("\nResampling dataframe:")
df_resample = df_2021.resample("W-SUN").first()
print(df_resample)
print("\nResampling series for OHLC:")
df_ohlc = df_2021["San Antonio Interstate 35"].resample("W-SUN").ohlc()
print(df_ohlc)
print("\nResampling series with forward fill:")
series_ffill = df_2021["San Antonio Interstate 35"].resample("H").ffill()
print(series_ffill)

# rolling
print("\nRolling mean:")
df_mean = df_2021["San Antonio Interstate 35"].rolling(10).mean()
print(df_mean)

# Plot moving average
fig = plt.figure(figsize=(12,6))
plt.plot(df_2021["San Antonio Interstate 35"], label="daily")
plt.plot(df_2021["San Antonio Interstate 35"].rolling(10, min_periods=5).mean(),
         label="10-day MA")
plt.legend()
plt.ylabel("PM 2.5")
plt.show()
```

---

### Chapter Summary / 章节总结

# Chapter 28 Summary / 第28章总结

## Theme / 主题: Chapter 28 / Chapter 28

This chapter contains **20 code files** demonstrating chapter 28.

本章包含 **20 个代码文件**，演示Chapter 28。

---
## Evolution / 演化路线

  1. `01_readexcel.ipynb` — Readexcel
  2. `02_info.ipynb` — Info
  3. `03_series.ipynb` — Series
  4. `04_fancy.ipynb` — Fancy
  5. `05_column.ipynb` — Column
  6. `06_unique.ipynb` — Unique
  7. `07_mean.ipynb` — Mean
  8. `08_describe.ipynb` — Describe
  9. `09_boolean.ipynb` — Boolean
  10. `10_highway.ipynb` — Highway
  11. `11_iloc.ipynb` — Iloc
  12. `13_pivot.ipynb` — Pivot
  13. `14_melt.ipynb` — Melt
  14. `16_groupby.ipynb` — Groupby
  15. `17_table.ipynb` — Table
  16. `18_join.ipynb` — Join
  17. `19_merge.ipynb` — Merge
  18. `20_apply.ipynb` — Apply
  19. `21_pandas.ipynb` — Pandas
  20. `33_timeseries.ipynb` — Timeseries

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 28) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 28）是机器学习流水线中的基础构建块。

---
