# 数据科学入门 / Beginner's Guide to Data Science
## Chapter 06

---

### Geodataframe

# 03 — Geodataframe / 03 Geodataframe

**Chapter 06 — File 1 of 2 / 第06章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Load the dataset**.

本脚本演示 **Load the dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
```

---
## Step 2 — Load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 3 — Convert the DataFrame to a GeoDataFrame

```python
# 将多个序列配对 / Pair multiple sequences
geometry = [Point(xy) for xy in zip(Ames['Longitude'], Ames['Latitude'])]
geo_df = gpd.GeoDataFrame(Ames, geometry=geometry)
# 打印输出 / Print output
print(geo_df)
```

---
## Learning Notes / 学习笔记

- **概念**: Load the dataset 是机器学习中的常用技术。  
  *Load the dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Geodataframe / 03 Geodataframe
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Convert the DataFrame to a GeoDataFrame
# 将多个序列配对 / Pair multiple sequences
geometry = [Point(xy) for xy in zip(Ames['Longitude'], Ames['Latitude'])]
geo_df = gpd.GeoDataFrame(Ames, geometry=geometry)
# 打印输出 / Print output
print(geo_df)
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Plotmap

# 07 — Plotmap / 07 Plotmap

**Chapter 06 — File 2 of 2 / 第06章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Convert the DataFrame to a GeoDataFrame**.

本脚本演示 **Convert the DataFrame to a GeoDataFrame**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
import geopandas as gpd
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Convert the DataFrame to a GeoDataFrame

```python
# 将多个序列配对 / Pair multiple sequences
geometry = [Point(xy) for xy in zip(Ames['Longitude'], Ames['Latitude'])]
geo_df = gpd.GeoDataFrame(Ames, geometry=geometry)
```

---
## Step 3 — Create a convex hull around the points

```python
convex_hull = geo_df.unary_union.convex_hull
convex_hull_geo = gpd.GeoSeries(convex_hull, crs="EPSG:4326")
convex_hull_transformed = convex_hull_geo.to_crs(epsg=3857)
buffered_hull = convex_hull_transformed.buffer(500)
```

---
## Step 4 — Plotting the map with Sale Prices, a basemap, and the buffered convex hull as a border

```python
fig, ax = plt.subplots(figsize=(12, 8))
geo_df.set_crs(epsg=4326).to_crs(epsg=3857) \
      .plot(column='SalePrice', cmap='coolwarm', ax=ax, legend=True, markersize=20)
buffered_hull.boundary.plot(ax=ax, color='black', label='Buffered Boundary of Ames')
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.set_axis_off()
ax.legend(loc='upper right')
colorbar = ax.get_figure().get_axes()[1]
colorbar.set_ylabel('Sale Price', rotation=270, labelpad=20, fontsize=15)
# 设置图表标题 / Set chart title
plt.title('Sales Prices of Individual Houses in Ames, Iowa with Buffered Boundary',
          fontsize=18)
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Convert the DataFrame to a GeoDataFrame 是机器学习中的常用技术。  
  *Convert the DataFrame to a GeoDataFrame is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plotmap / 07 Plotmap
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import geopandas as gpd
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Convert the DataFrame to a GeoDataFrame
# 将多个序列配对 / Pair multiple sequences
geometry = [Point(xy) for xy in zip(Ames['Longitude'], Ames['Latitude'])]
geo_df = gpd.GeoDataFrame(Ames, geometry=geometry)

# Create a convex hull around the points
convex_hull = geo_df.unary_union.convex_hull
convex_hull_geo = gpd.GeoSeries(convex_hull, crs="EPSG:4326")
convex_hull_transformed = convex_hull_geo.to_crs(epsg=3857)
buffered_hull = convex_hull_transformed.buffer(500)

# Plotting the map with Sale Prices, a basemap, and the buffered convex hull as a border
fig, ax = plt.subplots(figsize=(12, 8))
geo_df.set_crs(epsg=4326).to_crs(epsg=3857) \
      .plot(column='SalePrice', cmap='coolwarm', ax=ax, legend=True, markersize=20)
buffered_hull.boundary.plot(ax=ax, color='black', label='Buffered Boundary of Ames')
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.set_axis_off()
ax.legend(loc='upper right')
colorbar = ax.get_figure().get_axes()[1]
colorbar.set_ylabel('Sale Price', rotation=270, labelpad=20, fontsize=15)
# 设置图表标题 / Set chart title
plt.title('Sales Prices of Individual Houses in Ames, Iowa with Buffered Boundary',
          fontsize=18)
# 显示图表 / Display the plot
plt.show()
```

---

### Chapter Summary / 章节总结

# Chapter 06 Summary / 第06章总结

## Theme / 主题: Chapter 06 / Chapter 06

This chapter contains **2 code files** demonstrating chapter 06.

本章包含 **2 个代码文件**，演示Chapter 06。

---
## Evolution / 演化路线

  1. `03_geodataframe.ipynb` — Geodataframe
  2. `07_plotmap.ipynb` — Plotmap

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 06) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 06）是机器学习流水线中的基础构建块。

---
