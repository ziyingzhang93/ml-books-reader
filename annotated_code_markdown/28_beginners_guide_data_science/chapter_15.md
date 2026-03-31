# 数据科学入门 / Beginner's Guide to Data Science
## Chapter 15

---

### Top

# 02 — Top / 02 Top

**Chapter 15 — File 1 of 6 / 第15章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Identify the 10 most expensive homes based on SalePrice with key features**.

本脚本演示 **Identify the 10 most expensive homes based on SalePrice with key features**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Identify the 10 most expensive homes based on SalePrice with key features

```python
top_10_expensive_homes = Ames.nlargest(10, 'SalePrice')
features = ['SalePrice', 'GrLivArea', 'OverallQual', 'KitchenQual', 'TotRmsAbvGrd',
            'Fireplaces']
top_10_df = top_10_expensive_homes[features]
# 打印输出 / Print output
print(top_10_df)
```

---
## Learning Notes / 学习笔记

- **概念**: Identify the 10 most expensive homes based on SalePrice with key features 是机器学习中的常用技术。  
  *Identify the 10 most expensive homes based on SalePrice with key features is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Top / 02 Top
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Identify the 10 most expensive homes based on SalePrice with key features
top_10_expensive_homes = Ames.nlargest(10, 'SalePrice')
features = ['SalePrice', 'GrLivArea', 'OverallQual', 'KitchenQual', 'TotRmsAbvGrd',
            'Fireplaces']
top_10_df = top_10_expensive_homes[features]
# 打印输出 / Print output
print(top_10_df)
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Highquality

# 03 — Highquality / 03 Highquality

**Chapter 15 — File 2 of 6 / 第15章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Refine the search with highest quality, excellent kitchen, and 2 fireplaces**.

本脚本演示 **Refine the search with highest quality, excellent kitchen, and 2 fireplaces**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
top_10_expensive_homes = Ames.nlargest(10, 'SalePrice')
features = ['SalePrice', 'GrLivArea', 'OverallQual', 'KitchenQual', 'TotRmsAbvGrd',
            'Fireplaces']
top_10_df = top_10_expensive_homes[features]
```

---
## Step 2 — Refine the search with highest quality, excellent kitchen, and 2 fireplaces

```python
elite = top_10_df.query('OverallQual == 10 & KitchenQual == "Ex" & Fireplaces >= 2')
# 打印输出 / Print output
print(elite)
```

---
## Learning Notes / 学习笔记

- **概念**: Refine the search with highest quality, excellent kitchen, and 2 fireplaces 是机器学习中的常用技术。  
  *Refine the search with highest quality, excellent kitchen, and 2 fireplaces is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Highquality / 03 Highquality
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
top_10_expensive_homes = Ames.nlargest(10, 'SalePrice')
features = ['SalePrice', 'GrLivArea', 'OverallQual', 'KitchenQual', 'TotRmsAbvGrd',
            'Fireplaces']
top_10_df = top_10_expensive_homes[features]

# Refine the search with highest quality, excellent kitchen, and 2 fireplaces
elite = top_10_df.query('OverallQual == 10 & KitchenQual == "Ex" & Fireplaces >= 2')
# 打印输出 / Print output
print(elite)
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Psf

# 04 — Psf / 04 Psf

**Chapter 15 — File 3 of 6 / 第15章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Introduce PSF to rank the options**.

本脚本演示 **Introduce PSF to rank the options**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
top_10_expensive_homes = Ames.nlargest(10, 'SalePrice')
features = ['SalePrice', 'GrLivArea', 'OverallQual', 'KitchenQual', 'TotRmsAbvGrd',
            'Fireplaces']
top_10_df = top_10_expensive_homes[features]
elite = top_10_df.query('OverallQual == 10 & KitchenQual == "Ex" & Fireplaces >= 2') \
                 .copy()
```

---
## Step 2 — Introduce PSF to rank the options

```python
elite['PSF'] = elite['SalePrice']/elite['GrLivArea']
# 打印输出 / Print output
print(elite.sort_values(by='PSF'))
```

---
## Learning Notes / 学习笔记

- **概念**: Introduce PSF to rank the options 是机器学习中的常用技术。  
  *Introduce PSF to rank the options is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Psf / 04 Psf
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
top_10_expensive_homes = Ames.nlargest(10, 'SalePrice')
features = ['SalePrice', 'GrLivArea', 'OverallQual', 'KitchenQual', 'TotRmsAbvGrd',
            'Fireplaces']
top_10_df = top_10_expensive_homes[features]
elite = top_10_df.query('OverallQual == 10 & KitchenQual == "Ex" & Fireplaces >= 2') \
                 .copy()

# Introduce PSF to rank the options
elite['PSF'] = elite['SalePrice']/elite['GrLivArea']
# 打印输出 / Print output
print(elite.sort_values(by='PSF'))
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Bestvalue

# 05 — Bestvalue / 05 Bestvalue

**Chapter 15 — File 4 of 6 / 第15章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Cross check entire homes to search for better value**.

本脚本演示 **Cross check entire homes to search for better value**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Cross check entire homes to search for better value

```python
Ames['PSF'] = Ames['SalePrice']/Ames['GrLivArea']
value = Ames.query('PSF < 175 & OverallQual == 10 & KitchenQual == "Ex" & Fireplaces >=2')
# 打印输出 / Print output
print(value[['SalePrice', 'GrLivArea', 'OverallQual', 'KitchenQual', 'TotRmsAbvGrd',
             'Fireplaces', 'PSF']])
```

---
## Learning Notes / 学习笔记

- **概念**: Cross check entire homes to search for better value 是机器学习中的常用技术。  
  *Cross check entire homes to search for better value is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Bestvalue / 05 Bestvalue
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Cross check entire homes to search for better value
Ames['PSF'] = Ames['SalePrice']/Ames['GrLivArea']
value = Ames.query('PSF < 175 & OverallQual == 10 & KitchenQual == "Ex" & Fireplaces >=2')
# 打印输出 / Print output
print(value[['SalePrice', 'GrLivArea', 'OverallQual', 'KitchenQual', 'TotRmsAbvGrd',
             'Fireplaces', 'PSF']])
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Map

# 06 — Map / 06 Map

**Chapter 15 — File 5 of 6 / 第15章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Filter the dataset for these observations to get their latitude and longitude**.

本脚本演示 **Filter the dataset for these observations to get their latitude and longitude**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import folium

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
Ames['PSF'] = Ames['SalePrice']/Ames['GrLivArea']
value = Ames.query('PSF < 175 & OverallQual == 10 & KitchenQual == "Ex" & Fireplaces >=2')

final_observation_indexes = value.index.tolist()
```

---
## Step 2 — Filter the dataset for these observations to get their latitude and longitude

```python
final_locations = Ames.loc[final_observation_indexes, ['Latitude', 'Longitude']]
```

---
## Step 3 — Create a Folium map centered around the average location of the final observations

```python
map_center = [final_locations['Latitude'].mean(), final_locations['Longitude'].mean()]
value_map = folium.Map(location=map_center, zoom_start=12)
```

---
## Step 4 — Add information to markers

```python
for idx, row in final_locations.iterrows():
```

---
## Step 5 — Extract additional information for the popup

```python
info = value.loc[idx, ['SalePrice', 'GrLivArea', 'OverallQual', 'KitchenQual',
                           'TotRmsAbvGrd', 'Fireplaces', 'PSF']]
    popup_text = f"""<b>Index:</b> {idx}<br>
                     <b>SalePrice:</b> {info['SalePrice']}<br>
                     <b>GrLivArea:</b> {info['GrLivArea']} sqft<br>
                     <b>OverallQual:</b> {info['OverallQual']}<br>
                     <b>KitchenQual:</b> {info['KitchenQual']}<br>
                     <b>TotRmsAbvGrd:</b> {info['TotRmsAbvGrd']}<br>
                     <b>Fireplaces:</b> {info['Fireplaces']}<br>
                     <b>PSF:</b> ${info['PSF']:.2f} /sqft"""
    folium.Marker([row['Latitude'], row['Longitude']],
                  popup=folium.Popup(popup_text, max_width=250)).add_to(value_map)
```

---
## Step 6 — Save the map to an HTML file on working directory

```python
value_map.save('value_map.html')
```

---
## Learning Notes / 学习笔记

- **概念**: Filter the dataset for these observations to get their latitude and longitude 是机器学习中的常用技术。  
  *Filter the dataset for these observations to get their latitude and longitude is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Map / 06 Map
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import folium

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
Ames['PSF'] = Ames['SalePrice']/Ames['GrLivArea']
value = Ames.query('PSF < 175 & OverallQual == 10 & KitchenQual == "Ex" & Fireplaces >=2')

final_observation_indexes = value.index.tolist()

# Filter the dataset for these observations to get their latitude and longitude
final_locations = Ames.loc[final_observation_indexes, ['Latitude', 'Longitude']]

# Create a Folium map centered around the average location of the final observations
map_center = [final_locations['Latitude'].mean(), final_locations['Longitude'].mean()]
value_map = folium.Map(location=map_center, zoom_start=12)

# Add information to markers
for idx, row in final_locations.iterrows():
    # Extract additional information for the popup
    info = value.loc[idx, ['SalePrice', 'GrLivArea', 'OverallQual', 'KitchenQual',
                           'TotRmsAbvGrd', 'Fireplaces', 'PSF']]
    popup_text = f"""<b>Index:</b> {idx}<br>
                     <b>SalePrice:</b> {info['SalePrice']}<br>
                     <b>GrLivArea:</b> {info['GrLivArea']} sqft<br>
                     <b>OverallQual:</b> {info['OverallQual']}<br>
                     <b>KitchenQual:</b> {info['KitchenQual']}<br>
                     <b>TotRmsAbvGrd:</b> {info['TotRmsAbvGrd']}<br>
                     <b>Fireplaces:</b> {info['Fireplaces']}<br>
                     <b>PSF:</b> ${info['PSF']:.2f} /sqft"""
    folium.Marker([row['Latitude'], row['Longitude']],
                  popup=folium.Popup(popup_text, max_width=250)).add_to(value_map)

# Save the map to an HTML file on working directory
value_map.save('value_map.html')
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Heatmap

# 07 — Heatmap / 07 Heatmap

**Chapter 15 — File 6 of 6 / 第15章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Filter out rows with NaN values in 'Latitude' or 'Longitude'**.

本脚本演示 **Filter out rows with NaN values in 'Latitude' or 'Longitude'**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import folium
from folium.plugins import HeatMap

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Filter out rows with NaN values in 'Latitude' or 'Longitude'

```python
# 删除含缺失值的行 / Drop rows with missing values
Ames_Heat = Ames.dropna(subset=['Latitude', 'Longitude'])
```

---
## Step 3 — Group by 'Neighborhood' and calculate mean 'Latitude' and 'Longitude'

```python
neighborhood_locs = Ames_Heat.groupby('Neighborhood') \
                             .agg({'Latitude':'mean', 'Longitude':'mean'}) \
                             .reset_index()
```

---
## Step 4 — Create a map centered around Ames, Iowa

```python
ames_map_center = [Ames_Heat['Latitude'].mean(), Ames_Heat['Longitude'].mean()]
ames_heatmap = folium.Map(location=ames_map_center, zoom_start=12)
```

---
## Step 5 — Extract latitude and longitude data for the heatmap

```python
# 将多个序列配对 / Pair multiple sequences
heat_data = [(lat,lon) for lat, lon in zip(Ames_Heat['Latitude'], Ames_Heat['Longitude'])]
```

---
## Step 6 — Create and add a HeatMap layer to the map

```python
HeatMap(heat_data, radius=12).add_to(ames_heatmap)
```

---
## Step 7 — Add one black flag per neighborhood to the map

```python
for index, row in neighborhood_locs.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=row['Neighborhood'],
        icon=folium.Icon(color='black', icon='flag')
    ).add_to(ames_heatmap)
```

---
## Step 8 — Save the map to an HTML file in the working directory

```python
ames_heatmap.save('ames_heatmap.html')
```

---
## Learning Notes / 学习笔记

- **概念**: Filter out rows with NaN values in 'Latitude' or 'Longitude' 是机器学习中的常用技术。  
  *Filter out rows with NaN values in 'Latitude' or 'Longitude' is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `dropna` | 删除缺失值 | Drop missing values |
| `groupby` | 分组聚合 | Group and aggregate |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Heatmap / 07 Heatmap
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import folium
from folium.plugins import HeatMap

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Filter out rows with NaN values in 'Latitude' or 'Longitude'
# 删除含缺失值的行 / Drop rows with missing values
Ames_Heat = Ames.dropna(subset=['Latitude', 'Longitude'])

# Group by 'Neighborhood' and calculate mean 'Latitude' and 'Longitude'
neighborhood_locs = Ames_Heat.groupby('Neighborhood') \
                             .agg({'Latitude':'mean', 'Longitude':'mean'}) \
                             .reset_index()

# Create a map centered around Ames, Iowa
ames_map_center = [Ames_Heat['Latitude'].mean(), Ames_Heat['Longitude'].mean()]
ames_heatmap = folium.Map(location=ames_map_center, zoom_start=12)

# Extract latitude and longitude data for the heatmap
# 将多个序列配对 / Pair multiple sequences
heat_data = [(lat,lon) for lat, lon in zip(Ames_Heat['Latitude'], Ames_Heat['Longitude'])]

# Create and add a HeatMap layer to the map
HeatMap(heat_data, radius=12).add_to(ames_heatmap)

# Add one black flag per neighborhood to the map
for index, row in neighborhood_locs.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=row['Neighborhood'],
        icon=folium.Icon(color='black', icon='flag')
    ).add_to(ames_heatmap)

# Save the map to an HTML file in the working directory
ames_heatmap.save('ames_heatmap.html')
```

---

### Chapter Summary / 章节总结

# Chapter 15 Summary / 第15章总结

## Theme / 主题: Chapter 15 / Chapter 15

This chapter contains **6 code files** demonstrating chapter 15.

本章包含 **6 个代码文件**，演示Chapter 15。

---
## Evolution / 演化路线

  1. `02_top.ipynb` — Top
  2. `03_highquality.ipynb` — Highquality
  3. `04_psf.ipynb` — Psf
  4. `05_bestvalue.ipynb` — Bestvalue
  5. `06_map.ipynb` — Map
  6. `07_heatmap.ipynb` — Heatmap

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 15) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 15）是机器学习流水线中的基础构建块。

---
