# 线性代数与机器学习
## Chapter 22

---

### Download Data

# 22.1 — Download World Bank Data / 下载世界银行数据

**Chapter 22 — File 1 of 2 / 第22章 — 第1个文件（共2个）**

## Summary / 总结

Download economic indicators from the World Bank API for country comparison analysis. This demonstrates data collection from real-world sources for machine learning applications.

从世界银行API下载经济指标用于国家比较分析。这演示了从真实来源收集机器学习数据。

## Data Description / 数据描述

World Bank indicators for 2010:
- NE.EXP.GNFS.CD: Exports of goods and services (current US$)
- NE.IMP.GNFS.CD: Imports of goods and services (current US$)
- NV.AGR.TOTL.CD: Agriculture, value added (current US$)
- NY.GDP.MKTP.CD: GDP (current US$)
- NE.RSB.GNFS.CD: External balance on goods and services (current US$)

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


## Step 1 — Import Libraries / 导入库

```python
# Import data download library and pandas
# pandas_datareader requires internet connection
# 导入数据下载库和pandas
# pandas_datareader需要互联网连接
from pandas_datareader import wb
import pandas as pd
```

## Step 2 — Define Indicators / 定义指标

```python
# Define World Bank indicator codes to download
# 定义要下载的世界银行指标代码
indicator_names = [
    "NE.EXP.GNFS.CD",  # Exports of goods and services (current US$)
    "NE.IMP.GNFS.CD",  # Imports of goods and services (current US$)
    "NV.AGR.TOTL.CD",  # Agriculture, value added (current US$)
    "NY.GDP.MKTP.CD",  # GDP (current US$)
    "NE.RSB.GNFS.CD",  # External balance on goods and services (current US$)
]

print(f"Downloading {len(indicator_names)} economic indicators from World Bank...")
print(f"Note: This requires internet connection")
```

## Step 3 — Download World Bank Data / 下载世界银行数据

```python
# Download all countries' data for the indicators in 2010
# 下载2010年所有国家的指标数据
try:
    df = wb.download(
        country='all',              # Download for all countries
        indicator=indicator_names,  # These indicators
        start=2010,                # Year 2010
        end=2010                   # Single year
    ).reset_index()
    print(f"\nData downloaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"\nError downloading data: {e}")
    print(f"Note: This requires internet connection to World Bank API")
```

## Step 4 — Get Non-Aggregated Countries / 获取非汇总国家

```python
# Filter out aggregate regions (World, income groups, etc.)
# Keep only individual countries
# 过滤汇总地区（世界、收入组等）
# 仅保留单个国家
countries = wb.get_countries()
print(f"\nTotal countries/regions in WB: {len(countries)}")

# Get only non-aggregate entries
non_aggregates = countries[countries['region'] != 'Aggregates'].name
print(f"Individual countries: {len(non_aggregates)}")

# Filter dataframe to include only individual countries
df_nonagg = df[df['country'].isin(non_aggregates)].dropna()
print(f"\nFiltered dataset shape: {df_nonagg.shape}")
print(f"Countries with complete data: {df_nonagg['country'].nunique()}")
```

## Step 5 — Display Sample Data / 显示样本数据

```python
# Show first few rows of data
# 显示数据的前几行
print(f"\nFirst 5 rows:")
print(df_nonagg.head())

print(f"\nData types:")
print(df_nonagg.dtypes)

print(f"\nData statistics:")
print(df_nonagg.describe())
```

```python
## Step 6 — Check for Missing Values / 检查缺失值
```

```python
# Check missing values
# 检查缺失值
print(f"\nMissing values per column:")
print(df_nonagg.isnull().sum())

print(f"\nMissing percentage:")
missing_pct = (df_nonagg.isnull().sum() / len(df_nonagg) * 100)
for col, pct in missing_pct.items():
    print(f"  {col}: {pct:.1f}%")
```

## Learning Notes / 学习笔记

- **Data Collection**: World Bank data is cleaned and publicly available. Real-world data often has missing values (some countries don't report all indicators). Data cleaning and handling missing values is essential before analysis.
  
  **数据收集**：世界银行数据是清理过的公开数据。真实世界数据通常有缺失值。在分析前，数据清理和处理缺失值至关重要。

- **ML Application**: (1) Understanding data structure and completeness before analysis, (2) Filtering aggregate regions preserves individual country data, (3) Economic indicators show different scales (GDP is much larger than agriculture %), which highlights importance of feature scaling before distance-based algorithms.
  
  **ML应用**：(1) 在分析前理解数据结构和完整性，(2) 过滤汇总地区保留个别国家数据，(3) 经济指标显示不同尺度，强调了特征缩放的重要性。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `describe()` | 统计摘要信息 | Statistical summary |
| `dropna` | 删除缺失值 | Drop missing values |
| `head()` | 查看前几行数据 | View first few rows |
| `pandas` | 数据分析库 | Data analysis library |

➡️ **Next / 下一步**: `02_compare_countries.ipynb` — Computing distances between countries

## Complete Code / 完整代码一览

```python
# --- Import Section / 导入部分 ---
from pandas_datareader import wb
import pandas as pd

# --- Define Indicators / 定义指标 ---
names = [
    "NE.EXP.GNFS.CD",
    "NE.IMP.GNFS.CD",
    "NV.AGR.TOTL.CD",
    "NY.GDP.MKTP.CD",
    "NE.RSB.GNFS.CD",
]

# --- Download Data / 下载数据 ---
df = wb.download(
    country='all',
    indicator=names,
    start=2010,
    end=2010
).reset_index()

# --- Filter Countries / 过滤国家 ---
countries = wb.get_countries()
non_aggregates = countries[countries['region'] != 'Aggregates'].name
df_nonagg = df[df['country'].isin(non_aggregates)].dropna()

# --- Display Results / 显示结果 ---
print(df_nonagg)
```

---

### Compare Countries

# 22.2 — Compare Countries Using Distance Metrics / 使用距离度量比较国家

**Chapter 22 — File 2 of 2 / 第22章 — 第2个文件（共2个）**

## Summary / 总结

Compute Euclidean and cosine distances between countries based on economic indicators. Demonstrates how vector distances can be used to find similar countries and analyze economic relationships.

基于经济指标计算国家之间的欧几里得距离和余弦距离。演示如何使用向量距离来查找相似国家和分析经济关系。

## Distance Metrics / 距离度量

1. **Euclidean Distance**: $d(x,y) = \sqrt{\sum (x_i - y_i)^2}$ (raw difference)
2. **Cosine Similarity**: $\cos(\theta) = \frac{x \cdot y}{||x|| ||y||}$ (relative direction)

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


## Step 1 — Import Libraries / 导入库

```python
# Import necessary libraries
# 导入必要的库
from pandas_datareader import wb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
```

## Step 2 — Download and Prepare Data / 下载和准备数据

```python
# Download World Bank data
# 下载世界银行数据
names = [
    "NE.EXP.GNFS.CD",  # Exports
    "NE.IMP.GNFS.CD",  # Imports
    "NV.AGR.TOTL.CD",  # Agriculture
    "NY.GDP.MKTP.CD",  # GDP
    "NE.RSB.GNFS.CD",  # External balance
]

try:
    df = wb.download(
        country='all',
        indicator=names,
        start=2010,
        end=2010
    ).reset_index()
    
    # Filter non-aggregate countries
    countries = wb.get_countries()
    non_aggregates = countries[countries['region'] != 'Aggregates'].name
    df = df[df['country'].isin(non_aggregates)].dropna()
    
    print(f"Data loaded: {df.shape}")
except Exception as e:
    print(f"Note: Could not download data: {e}")
    print(f"This notebook requires internet connection to World Bank API")
```

## Step 3 — Prepare Feature Matrix / 准备特征矩阵

```python
# Extract feature matrix and country names
# 提取特征矩阵和国家名称
X = df.iloc[:, 2:].values  # Exclude year and country columns
country_names = df['country'].values

print(f"Feature matrix shape: {X.shape}")
print(f"Number of countries: {len(country_names)}")
print(f"Number of features: {X.shape[1]}")

# Show first few countries
print(f"\nFirst 5 countries: {country_names[:5]}")
```

## Step 4 — Scale Features / 缩放特征

```python
# Standardize features (important for distance metrics)
# 标准化特征（对距离度量很重要）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Scaled feature matrix shape: {X_scaled.shape}")
print(f"Mean of scaled features: {X_scaled.mean(axis=0)}")
print(f"Std of scaled features: {X_scaled.std(axis=0)}")
```

## Step 5 — Compute Euclidean Distances / 计算欧几里得距离

```python
# Compute pairwise Euclidean distances
# 计算两两欧几里得距离
euclidean_dist = euclidean_distances(X_scaled)

print(f"Euclidean distance matrix shape: {euclidean_dist.shape}")
print(f"\nDistance statistics:")
print(f"  Min distance: {euclidean_dist[euclidean_dist > 0].min():.4f}")
print(f"  Max distance: {euclidean_dist.max():.4f}")
print(f"  Mean distance: {euclidean_dist[euclidean_dist > 0].mean():.4f}")
```

## Step 6 — Compute Cosine Distances / 计算余弦距离

```python
# Compute cosine distances (1 - cosine similarity)
# 计算余弦距离
cosine_dist = cosine_distances(X_scaled)

print(f"Cosine distance matrix shape: {cosine_dist.shape}")
print(f"\nDistance statistics:")
print(f"  Min distance: {cosine_dist[cosine_dist > 0].min():.4f}")
print(f"  Max distance: {cosine_dist.max():.4f}")
print(f"  Mean distance: {cosine_dist[cosine_dist > 0].mean():.4f}")
```

## Step 7 — Find Similar Countries to Australia / 查找与澳大利亚相似的国家

```python
# Find index of a specific country (Australia)
# 查找特定国家（澳大利亚）的索引
try:
    australia_idx = np.where(country_names == 'Australia')[0][0]
    
    # Get distances from Australia to all other countries
    # 获取从澳大利亚到所有其他国家的距离
    distances_to_australia_euclidean = euclidean_dist[australia_idx]
    distances_to_australia_cosine = cosine_dist[australia_idx]
    
    # Find 10 closest countries (excluding Australia itself)
    # 查找10个最近的国家（排除澳大利亚本身）
    n_closest = 10
    
    # Euclidean distances
    euclidean_closest = np.argsort(distances_to_australia_euclidean)[1:n_closest+1]
    print("Countries most similar to Australia (by Euclidean distance):")
    for i, idx in enumerate(euclidean_closest, 1):
        print(f"  {i}. {country_names[idx]}: {distances_to_australia_euclidean[idx]:.4f}")
    
    # Cosine distances
    cosine_closest = np.argsort(distances_to_australia_cosine)[1:n_closest+1]
    print(f"\nCountries most similar to Australia (by Cosine distance):")
    for i, idx in enumerate(cosine_closest, 1):
        print(f"  {i}. {country_names[idx]}: {distances_to_australia_cosine[idx]:.4f}")
        
except Exception as e:
    print(f"Could not find Australia: {e}")
```

## Step 8 — Compare Distance Metrics / 比较距离度量

```python
# Discuss differences between Euclidean and Cosine distances
# 讨论欧几里得距离和余弦距离的差异
print(f"\nDistance Metric Comparison:")
print(f"\nEuclidean Distance:")
print(f"  - Measures absolute differences between vectors")
print(f"  - Sensitive to scale of features")
print(f"  - Geometric distance in feature space")
print(f"\nCosine Distance:")
print(f"  - Measures angle between vectors")
print(f"  - Invariant to scale (measures direction only)")
print(f"  - Better for comparing vector direction/pattern")
print(f"\nFor economic data:")
print(f"  - Euclidean: Countries with similar absolute sizes")
print(f"  - Cosine: Countries with similar economic structure/proportions")
```

## Learning Notes / 学习笔记

- **Math Essence**: Euclidean distance measures absolute differences (affected by scale), while cosine distance measures angular similarity (scale-invariant). Choosing the right metric depends on whether you care about absolute magnitude or relative direction.
  
  **数学本质**：欧几里得距离衡量绝对差异（受尺度影响），而余弦距离衡量角度相似性（尺度不变）。选择正确的度量取决于是否关心绝对大小或相对方向。

- **ML Application**: (1) Distance metrics are fundamental to clustering, nearest-neighbors, and similarity search, (2) Feature scaling is crucial for distance-based algorithms - StandardScaler normalizes each feature to mean=0, std=1, ensuring equal importance, (3) Different metrics can produce different results (see Australia example), choose based on domain knowledge of what "similarity" means for your data.
  
  **ML应用**：(1) 距离度量对聚类、最近邻和相似性搜索至关重要，(2) 特征缩放对基于距离的算法至关重要，(3) 不同度量可产生不同结果，根据领域知识选择。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `dropna` | 删除缺失值 | Drop missing values |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |

➡️ **Next / 下一步**: `../chapter_23/01_show_tar.ipynb` — Working with compressed tar archives

## Complete Code / 完整代码一览

```python
# --- Import Section / 导入部分 ---
from pandas_datareader import wb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

# --- Download Data / 下载数据 ---
names = [
    "NE.EXP.GNFS.CD",
    "NE.IMP.GNFS.CD",
    "NV.AGR.TOTL.CD",
    "NY.GDP.MKTP.CD",
    "NE.RSB.GNFS.CD",
]
df = wb.download(country='all', indicator=names, start=2010, end=2010).reset_index()
countries = wb.get_countries()
non_aggregates = countries[countries['region'] != 'Aggregates'].name
df = df[df['country'].isin(non_aggregates)].dropna()

# --- Prepare Features / 准备特征 ---
X = df.iloc[:, 2:].values
country_names = df['country'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Compute Distances / 计算距离 ---
euclidean_dist = euclidean_distances(X_scaled)
cosine_dist = cosine_distances(X_scaled)

# --- Find Similar Countries / 查找相似国家 ---
australia_idx = np.where(country_names == 'Australia')[0][0]
closest = np.argsort(euclidean_dist[australia_idx])[1:11]
for idx in closest:
    print(f"{country_names[idx]}: {euclidean_dist[australia_idx, idx]:.4f}")
```

---

### Chapter Summary

# Chapter 22 Summary / 第22章总结：Country Comparison via Vector Norms

## Theme / 主题

Vector norms and distance metrics measure similarity between samples. This chapter uses real World Bank data to compare countries based on economic indicators. By computing Euclidean distance (L2 norm) and cosine similarity (normalized dot product), we discover which countries are "similar" in economic profile. This is a practical application showing how linear algebra connects to real-world insights.

向量范数和距离度量衡量样本之间的相似性。本章使用真实世界银行数据根据经济指标比较国家。通过计算欧几里得距离（L2范数）和余弦相似性（规范化点积），我们发现哪些国家在经济概况上"相似"。这是一个实际应用，展示线性代数如何与现实见解相关联。

## Evolution / 演化路线

```
01_download_data.ipynb
    └─ Download World Bank data (下载世界银行数据)
    
02_euclidean_distance.ipynb
    └─ Compute pairwise Euclidean distances (计算成对欧几里得距离)
       ||A - B||_2 measures economic "distance"
    
03_cosine_similarity.ipynb
    └─ Compute cosine similarity (计算余弦相似性)
       cos(θ) = (A·B) / (||A|| ||B||) measures direction alignment
```

## Progression Logic / 进度逻辑

Real-world application progresses through **data → metric → insight**:

1. **Data**: Download actual World Bank economic indicators
2. **Euclidean distance**: ||A - B||_2
   - Measures overall difference (taking scale into account)
   - Small distance = similar economic profiles
3. **Cosine similarity**: cos(θ) = (A·B) / (||A|| ||B||)
   - Measures direction of vectors (ignoring magnitude)
   - High cosine = similar "shape" (proportional indicators)

The contrast is important: Euclidean distance cares about magnitude, cosine similarity cares about direction.

现实应用通过**数据→度量→见解**进行：

1. **数据**：下载实际世界银行经济指标
2. **欧几里得距离**：||A - B||_2
   - 测量总体差异（考虑规模）
   - 小距离=相似的经济概况
3. **余弦相似性**：cos(θ) = (A·B) / (||A|| ||B||)
   - 测量向量的方向（忽略幅度）
   - 高余弦=相似的"形状"（比例指标）

对比很重要：欧几里得距离关心幅度，余弦相似性关心方向。

## ML Relevance / 机器学习相关性

In machine learning:
- **Distance metrics**:
  - Euclidean: works best when features are on similar scale
  - Manhattan (L1): robust to outliers, encourages sparsity
  - Cosine: works best when magnitude is irrelevant (text, embeddings)

- **Similarity vs. Distance**:
  - Distance: larger = more different (Euclidean, Manhattan)
  - Similarity: larger = more similar (cosine, correlation)

- **When to use each**:
  - **Euclidean**: Default choice, general purpose
  - **Cosine**: Text (TF-IDF), embeddings (word2vec), images with normalization
  - **L1 (Manhattan)**: Sparse data, interpretability

- **k-NN algorithm**:
  - Find k nearest neighbors using distance metric
  - Decision: majority class of k neighbors
  - Euclidean is standard, cosine for high-dimensional sparse data

- **Clustering**:
  - k-means uses Euclidean distance
  - Determines which cluster each point belongs to
  - Different metrics → different clustering results

- **Recommendation systems**:
  - User similarity: cosine similarity of rating vectors
  - Item similarity: cosine similarity of feature vectors
  - Both leverage normalized dot product

- **Real-world challenge**:
  - Feature scaling: Euclidean distance sensitive to scale
  - Solution: normalize features to [0,1] or standardize (z-score)
  - With cosine: automatic scale invariance (normalized vectors)

- **World Bank example insights**:
  - Euclidean distance: rich vs. poor countries cluster (absolute values matter)
  - Cosine similarity: proportional economies cluster (ratios matter)
  - Example: UK and Germany might be similar in cosine (both developed)
  - But their GDP absolute values differ (Euclidean distance larger)

**Key insight**: Metric choice reveals different aspects of similarity. Euclidean is scale-dependent, cosine is scale-invariant. Choosing the right metric is crucial for distance-based ML algorithms.

在机器学习中：
- **距离度量**：
  - 欧几里得：当特征在相似规模上时效果最好
  - 曼哈顿（L1）：对异常值鲁棒，鼓励稀疏
  - 余弦：当幅度无关时效果最好（文本、嵌入）

- **相似性vs.距离**：
  - 距离：较大=更不同（欧几里得、曼哈顿）
  - 相似性：较大=更相似（余弦、相关性）

- **何时使用每个**：
  - **欧几里得**：默认选择、通用目的
  - **余弦**：文本（TF-IDF）、嵌入（word2vec）、规范化的图像
  - **L1（曼哈顿）**：稀疏数据、可解释性

- **k-NN算法**：
  - 使用距离度量找到k个最近邻
  - 决定：k个邻居的多数类
  - 欧几里得是标准的，对于高维稀疏数据使用余弦

- **聚类**：
  - k-means使用欧几里得距离
  - 确定每个点属于哪个聚类
  - 不同的度量→不同的聚类结果

- **推荐系统**：
  - 用户相似性：评分向量的余弦相似性
  - 项目相似性：特征向量的余弦相似性
  - 两者都利用规范化的点积

- **现实挑战**：
  - 特征缩放：欧几里得距离对规模敏感
  - 解决方案：将特征归一化到[0,1]或标准化（z分数）
  - 使用余弦：自动规模不变性（规范化向量）

- **世界银行示例见解**：
  - 欧几里得距离：富国vs.穷国聚类（绝对值重要）
  - 余弦相似性：比例经济聚类（比率重要）
  - 示例：英国和德国在余弦中可能相似（都是发达的）
  - 但它们的GDP绝对值不同（欧几里得距离更大）

**关键见解**：度量选择揭示相似性的不同方面。欧几里得是规模相关的，余弦是规模不变的。选择正确的度量对于基于距离的ML算法至关重要。

---
