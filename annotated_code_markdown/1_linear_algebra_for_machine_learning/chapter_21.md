# 线性代数与机器学习 / Linear Algebra for Machine Learning
## Chapter 21

---

### Load Data



---

### Visualize Wine

# 21.2 — Visualize Wine Dataset with PCA / 用PCA可视化葡萄酒数据集

**Chapter 21 — File 2 of 5 / 第21章 — 第2个文件（共5个）**

## Summary / 总结

Visualize the 13-dimensional wine dataset by projecting it onto 2D/3D using PCA. Demonstrates the importance of feature scaling and how PCA reveals class structure.

通过使用PCA投影将13维葡萄酒数据集可视化为2D/3D。演示特征缩放的重要性以及PCA如何揭示类别结构。

## Visualization Steps / 可视化步骤

1. Plot two features directly (reveals why we need PCA)
2. Plot three features in 3D space
3. Apply PCA without scaling
4. Apply PCA with StandardScaler (recommended)

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

## Step 1 — Import Libraries / 导入库

```python
# Import dataset, PCA, scaler, and visualization
# 导入数据集、PCA、缩放器和可视化
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import load_wine
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.decomposition import PCA
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
```

## Step 2 — Load Dataset / 加载数据集

```python
# Load wine dataset
# 加载葡萄酒数据集
winedata = load_wine()
X, y = winedata['data'], winedata['target']
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Dataset shape: {X.shape}")
# 打印输出 / Print output
print(f"Classes: {winedata['target_names']}")
```

## Step 3 — Visualization 1: Two Features Direct / 可视化1：直接两个特征

```python
# Plot two features directly to show raw data structure
# 绘制两个特征直接显示原始数据结构
# 创建画布 / Create figure canvas
plt.figure(figsize=(8, 6))
# 同时获取索引和值 / Get both index and value
for i, target in enumerate(set(y)):
    indices = y == target
    # 绘制散点图 / Draw scatter plot
    plt.scatter(X[indices, 1], X[indices, 2], 
               label=winedata['target_names'][target], alpha=0.7)
# 设置X轴标签 / Set X-axis label
plt.xlabel(winedata['feature_names'][1])
# 设置Y轴标签 / Set Y-axis label
plt.ylabel(winedata['feature_names'][2])
# 设置图表标题 / Set chart title
plt.title('Two Particular Features of Wine Dataset')
# 显示图例 / Show legend
plt.legend()
# 显示图表 / Display the plot
plt.show()
# 打印输出 / Print output
print("Note: Only two features show limited class separation")
```

## Step 4 — Visualization 2: Three Features in 3D / 可视化2：3D中的三个特征

```python
# Plot three features in 3D space
# 在3D空间中绘制三个特征
# 创建画布 / Create figure canvas
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 同时获取索引和值 / Get both index and value
for i, target in enumerate(set(y)):
    indices = y == target
    ax.scatter(X[indices, 1], X[indices, 2], X[indices, 3],
              label=winedata['target_names'][target], alpha=0.7)

ax.set_xlabel(winedata['feature_names'][1])
ax.set_ylabel(winedata['feature_names'][2])
ax.set_zlabel(winedata['feature_names'][3])
ax.set_title('Three Features in 3D Space')
ax.legend()
# 显示图表 / Display the plot
plt.show()
# 打印输出 / Print output
print("Note: Better separation but still not optimal")
```

## Step 5 — Visualization 3: PCA without Scaling / 可视化3：不缩放的PCA

```python
# Apply PCA directly without scaling
# This can be problematic if features have different scales
# 直接应用PCA而不缩放
# 主成分分析：降维，保留最重要的特征 / PCA: reduce dimensions, keep key features
pca = PCA(n_components=2)
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
Xt_unscaled = pca.fit_transform(X)

# 创建画布 / Create figure canvas
plt.figure(figsize=(8, 6))
# 同时获取索引和值 / Get both index and value
for i, target in enumerate(set(y)):
    indices = y == target
    # 绘制散点图 / Draw scatter plot
    plt.scatter(Xt_unscaled[indices, 0], Xt_unscaled[indices, 1],
               label=winedata['target_names'][target], alpha=0.7)

# 设置X轴标签 / Set X-axis label
plt.xlabel(f'PC1 (explained var: {pca.explained_variance_ratio_[0]:.3f})')
# 设置Y轴标签 / Set Y-axis label
plt.ylabel(f'PC2 (explained var: {pca.explained_variance_ratio_[1]:.3f})')
# 设置图表标题 / Set chart title
plt.title('PCA Projection (without scaling)')
# 显示图例 / Show legend
plt.legend()
# 显示图表 / Display the plot
plt.show()
# 打印输出 / Print output
print(f"Variance explained: {sum(pca.explained_variance_ratio_):.3f}")
```

## Step 6 — Visualization 4: PCA with Scaling (Recommended) / 可视化4：缩放后的PCA（推荐）

```python
# Apply PCA with StandardScaler preprocessing
# This is the recommended approach
# 使用StandardScaler预处理应用PCA
# 这是推荐的方法
# 主成分分析：降维，保留最重要的特征 / PCA: reduce dimensions, keep key features
pca = PCA(n_components=2)
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
Xt_scaled = pipe.fit_transform(X)

# 创建画布 / Create figure canvas
plt.figure(figsize=(8, 6))
# 绘制散点图 / Draw scatter plot
scatter = plt.scatter(Xt_scaled[:, 0], Xt_scaled[:, 1], c=y, 
                     cmap='viridis', s=100, alpha=0.7)

# Add color bar with class names
cbar = plt.colorbar(scatter)
cbar.set_label('Wine Class')

# 设置X轴标签 / Set X-axis label
plt.xlabel(f'PC1 (explained var: {pca.explained_variance_ratio_[0]:.3f})')
# 设置Y轴标签 / Set Y-axis label
plt.ylabel(f'PC2 (explained var: {pca.explained_variance_ratio_[1]:.3f})')
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
plt.title('PCA Projection (with StandardScaler - RECOMMENDED)')
# 显示图例 / Show legend
plt.legend(labels=winedata['target_names'])
# 显示图表 / Display the plot
plt.show()
# 打印输出 / Print output
print(f"Variance explained by PC1: {pca.explained_variance_ratio_[0]:.3f}")
# 打印输出 / Print output
print(f"Variance explained by PC2: {pca.explained_variance_ratio_[1]:.3f}")
# 打印输出 / Print output
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.3f}")
```

## Learning Notes / 学习笔记

- **Math Essence**: PCA finds the directions of maximum variance in the data. When features are on different scales, PCA is dominated by features with larger variance regardless of information content. StandardScaler ensures all features contribute equally to PCA.
  
  **数学本质**：PCA找到数据中最大方差的方向。当特征在不同尺度上时，PCA由具有较大方差的特征主导。StandardScaler确保所有特征对PCA有相等贡献。

- **ML Application**: (1) Always scale features before PCA unless you have domain knowledge suggesting otherwise, (2) The "elbow" in explained variance plot suggests optimal number of components, (3) PCA with 2 components captures ~60% of wine dataset variance, sufficient for visualization and downstream classification.
  
  **ML应用**：(1) 在PCA前始终缩放特征，(2) 解释方差图中的"肘部"表明最优成分数，(3) 2个成分的PCA捕获~60%的方差，足以进行可视化和分类。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `PCA` | 主成分分析，降维 | Principal Component Analysis, dimensionality reduction |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `matplotlib` | 绑图库 | Plotting library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.scatter` | 绘制散点图 | Draw scatter plot |
| `plt.show` | 显示图表 | Display plot |

➡️ **Next / 下一步**: `03_visualize_digits.ipynb` — Visualizing the handwritten digits dataset

## Complete Code / 完整代码一览

```python
# --- Import Section / 导入部分 ---
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import load_wine
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.decomposition import PCA
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Load Data / 加载数据 ---
winedata = load_wine()
X, y = winedata['data'], winedata['target']

# --- Plot 1: Two Features / 绘图1：两个特征 ---
# 创建画布 / Create figure canvas
plt.figure(figsize=(8, 6))
# 绘制散点图 / Draw scatter plot
plt.scatter(X[:, 1], X[:, 2], c=y)
# 设置X轴标签 / Set X-axis label
plt.xlabel(winedata['feature_names'][1])
# 设置Y轴标签 / Set Y-axis label
plt.ylabel(winedata['feature_names'][2])
# 设置图表标题 / Set chart title
plt.title('Two Features of Wine Dataset')
# 显示图表 / Display the plot
plt.show()

# --- Plot 4: PCA with Scaling / 绘图4：缩放后的PCA ---
# 主成分分析：降维，保留最重要的特征 / PCA: reduce dimensions, keep key features
pca = PCA(n_components=2)
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
Xt = pipe.fit_transform(X)
# 绘制散点图 / Draw scatter plot
plot = plt.scatter(Xt[:, 0], Xt[:, 1], c=y)
# 显示图例 / Show legend
plt.legend(handles=plot.legend_elements()[0], labels=list(winedata['target_names']))
# 设置X轴标签 / Set X-axis label
plt.xlabel('PC1')
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('PC2')
# 设置图表标题 / Set chart title
plt.title('First Two Principal Components (with Scaling)')
# 显示图表 / Display the plot
plt.show()
```

---

### Visualize Digits



---

### Visualize Iris

# 21.4 — Visualize Iris Dataset / 可视化鸢尾花数据集

**Chapter 21 — File 4 of 5 / 第21章 — 第4个文件（共5个）**

## Summary / 总结

Visualize the classic Iris dataset, one of the most famous datasets in machine learning. Shows that simple 2D visualization of 4D data can effectively separate classes.

可视化经典的鸢尾花数据集，机器学习中最著名的数据集之一。显示4D数据的简单2D可视化可以有效分离类别。

## Dataset Overview / 数据集概览

- **Samples**: 150 iris flowers
- **Features**: 4 (sepal length, sepal width, petal length, petal width)
- **Classes**: 3 iris species

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

## Step 1 — Import Libraries / 导入库

```python
# Import dataset loader and visualization
# 导入数据集加载器和可视化
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import load_iris
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
```

## Step 2 — Load Iris Dataset / 加载鸢尾花数据集

```python
# Load iris dataset
# 加载鸢尾花数据集
irisdata = load_iris()
X, y = irisdata['data'], irisdata['target']
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Dataset shape: {X.shape}")
# 打印输出 / Print output
print(f"Classes: {irisdata['target_names']}")
# 打印输出 / Print output
print(f"Features: {irisdata['feature_names']}")
```

## Step 3 — Display Data Statistics / 显示数据统计

```python
# Show class distribution
# 显示类别分布
# 打印输出 / Print output
print(f"\nClass distribution:")
# 同时获取索引和值 / Get both index and value
for i, name in enumerate(irisdata['target_names']):
    count = (y == i).sum()
    # 打印输出 / Print output
    print(f"  {name}: {count} samples")
```

## Step 4 — Simple 2D Visualization / 简单的2D可视化

```python
# Plot using first two features (sepal length, sepal width)
# 使用前两个特征绘图（萼片长度、萼片宽度）
# 创建画布 / Create figure canvas
plt.figure(figsize=(8, 6))

# Plot each class with a different color
# 用不同的颜色绘制每个类别
# 同时获取索引和值 / Get both index and value
for i, name in enumerate(irisdata['target_names']):
    indices = y == i
    # 绘制散点图 / Draw scatter plot
    plt.scatter(X[indices, 0], X[indices, 1], 
               label=name, alpha=0.7, s=100)

# 设置X轴标签 / Set X-axis label
plt.xlabel(irisdata['feature_names'][0])
# 设置Y轴标签 / Set Y-axis label
plt.ylabel(irisdata['feature_names'][1])
# 设置图表标题 / Set chart title
plt.title('Two Features from the Iris Dataset')
# 显示图例 / Show legend
plt.legend()
plt.grid(True, alpha=0.3)
# 显示图表 / Display the plot
plt.show()
# 打印输出 / Print output
print("Note: Iris Setosa is well separated, but other two classes overlap")
```

## Step 5 — Display First Few Samples / 显示前几个样本

```python
# Show first few samples with feature names
# 显示带特征名称的前几个样本
# 打印输出 / Print output
print(f"First 5 samples with class labels:")
# 生成整数序列 / Generate integer sequence
for i in range(5):
    features = X[i]
    label = irisdata['target_names'][y[i]]
    # 打印输出 / Print output
    print(f"\nSample {i}: {label}")
    # 同时获取索引和值 / Get both index and value
    for j, fname in enumerate(irisdata['feature_names']):
        # 打印输出 / Print output
        print(f"  {fname}: {features[j]:.2f}")
```

## Step 6 — Feature Statistics / 特征统计

```python
# Compute statistics for each feature
# 计算每个特征的统计
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 打印输出 / Print output
print(f"Feature statistics:")
# 同时获取索引和值 / Get both index and value
for i, fname in enumerate(irisdata['feature_names']):
    # 打印输出 / Print output
    print(f"\n{fname}:")
    # 打印输出 / Print output
    print(f"  Mean: {X[:, i].mean():.2f}")
    # 打印输出 / Print output
    print(f"  Std:  {X[:, i].std():.2f}")
    # 打印输出 / Print output
    print(f"  Min:  {X[:, i].min():.2f}")
    # 打印输出 / Print output
    print(f"  Max:  {X[:, i].max():.2f}")
```

## Learning Notes / 学习笔记

- **Math Essence**: The Iris dataset demonstrates that not all features are equally informative. Petal measurements separate classes much better than sepal measurements. This is a key insight for feature selection and dimensionality reduction.
  
  **数学本质**：鸢尾花数据集演示并非所有特征同等重要。花瓣测量比萼片测量更好地分离类别。这对特征选择和降维至关重要。

- **ML Application**: (1) The Iris dataset is often used as a first test for machine learning algorithms, (2) It shows that simple datasets can have predictable patterns, (3) Good for understanding PCA and feature importance - petal features contribute more than sepal features to class separation, (4) Despite being simple, it still has overlapping classes that make classification non-trivial.
  
  **ML应用**：(1) 鸢尾花数据集常用作机器学习算法的首次测试，(2) 显示简单数据集可以有可预测的模式，(3) 适合理解PCA和特征重要性，(4) 尽管简单，但仍具有使分类非平凡的重叠类别。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.scatter` | 绘制散点图 | Draw scatter plot |
| `plt.show` | 显示图表 | Display plot |

➡️ **Next / 下一步**: `05_iris_pca.ipynb` — PCA analysis with SVM on Iris dataset

## Complete Code / 完整代码一览

```python
# --- Import Section / 导入部分 ---
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import load_iris
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# --- Load Data / 加载数据 ---
irisdata = load_iris()
X, y = irisdata['data'], irisdata['target']

# --- Visualize / 可视化 ---
# 创建画布 / Create figure canvas
plt.figure(figsize=(8, 6))
# 绘制散点图 / Draw scatter plot
plt.scatter(X[:, 0], X[:, 1], c=y)
# 设置X轴标签 / Set X-axis label
plt.xlabel(irisdata['feature_names'][0])
# 设置Y轴标签 / Set Y-axis label
plt.ylabel(irisdata['feature_names'][1])
# 设置图表标题 / Set chart title
plt.title('Two Features from the Iris Dataset')
# 显示图表 / Display the plot
plt.show()
```

---

### Iris Pca

# 21.5 — PCA Analysis with SVM on Iris / 鸢尾花上的PCA和SVM分析

**Chapter 21 — File 5 of 5 / 第21章 — 第5个文件（共5个）**

## Summary / 总结

Apply PCA to the Iris dataset and compare classification performance using all features vs. using only principal components. Demonstrates how PCA reduces dimensionality while preserving classification ability.

对鸢尾花数据集应用PCA，比较使用所有特征与仅使用主成分的分类性能。演示PCA如何在保留分类能力的同时降低维度。

## Workflow / 工作流程

1. Load Iris data
2. Perform PCA to extract principal components
3. Visualize what each component captures
4. Remove components progressively
5. Compare SVM accuracy with all features vs. PC1 only

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
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

## Step 1 — Import Libraries / 导入库

```python
# Import all required libraries
# 导入所有必需的库
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import load_iris
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.decomposition import PCA
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import f1_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
```

## Step 2 — Load and Prepare Data / 加载和准备数据

```python
# Load iris dataset
# 加载鸢尾花数据集
irisdata = load_iris()
X, y = irisdata['data'], irisdata['target']
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Original data shape: {X.shape}")
# 打印输出 / Print output
print(f"Classes: {irisdata['target_names']}")
```

## Step 3 — Visualize Raw Data / 可视化原始数据

```python
# Plot two features to show raw data structure
# 绘制两个特征以显示原始数据结构
# 创建画布 / Create figure canvas
plt.figure(figsize=(8, 6))
# 同时获取索引和值 / Get both index and value
for i, name in enumerate(irisdata['target_names']):
    indices = y == i
    # 绘制散点图 / Draw scatter plot
    plt.scatter(X[indices, 0], X[indices, 1], label=name, alpha=0.7)
# 设置X轴标签 / Set X-axis label
plt.xlabel(irisdata['feature_names'][0])
# 设置Y轴标签 / Set Y-axis label
plt.ylabel(irisdata['feature_names'][1])
# 设置图表标题 / Set chart title
plt.title('Two Features from the Iris Dataset')
# 显示图例 / Show legend
plt.legend()
# 显示图表 / Display the plot
plt.show()
```

## Step 4 — Fit PCA and Analyze Components / 拟合PCA并分析成分

```python
# Perform PCA on full data (keeping all 4 components)
# 对全数据执行PCA（保留全部4个成分）
# 主成分分析：降维，保留最重要的特征 / PCA: reduce dimensions, keep key features
pca = PCA().fit(X)

# 打印输出 / Print output
print("Principal Components:")
# 打印输出 / Print output
print(pca.components_)

# 打印输出 / Print output
print(f"\nExplained Variance Ratio:")
# 同时获取索引和值 / Get both index and value
for i, ratio in enumerate(pca.explained_variance_ratio_):
    # 打印输出 / Print output
    print(f"  PC{i+1}: {ratio:.4f}")

# 打印输出 / Print output
print(f"\nCumulative Explained Variance:")
cumsum = 0
# 同时获取索引和值 / Get both index and value
for i, ratio in enumerate(pca.explained_variance_ratio_):
    cumsum += ratio
    # 打印输出 / Print output
    print(f"  PC{i+1}: {cumsum:.4f}")
```

## Step 5 — Visualize Component Importance / 可视化成分重要性

```python
# Plot explained variance
# 绘制解释方差
# 创建画布 / Create figure canvas
plt.figure(figsize=(10, 5))

# Plot 1: Variance explained by each component
# 创建子图 / Create subplot
plt.subplot(1, 2, 1)
# 绘制柱状图 / Draw bar chart
plt.bar(range(1, 5), pca.explained_variance_ratio_)
# 设置X轴标签 / Set X-axis label
plt.xlabel('Principal Component')
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('Explained Variance Ratio')
# 设置图表标题 / Set chart title
plt.title('Variance Explained by Each Component')
# 生成整数序列 / Generate integer sequence
plt.xticks(range(1, 5))

# Plot 2: Cumulative variance
# 创建子图 / Create subplot
plt.subplot(1, 2, 2)
cumsum = np.cumsum(pca.explained_variance_ratio_)
# 绘制折线图 / Draw line plot
plt.plot(range(1, 5), cumsum, 'b-o')
# 设置X轴标签 / Set X-axis label
plt.xlabel('Number of Components')
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('Cumulative Explained Variance')
# 设置图表标题 / Set chart title
plt.title('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
# 生成整数序列 / Generate integer sequence
plt.xticks(range(1, 5))
# 显示图例 / Show legend
plt.legend()
plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

## Step 6 — Analyze Feature Contributions / 分析特征贡献

```python
# Show how original features contribute to principal components
# 显示原始特征如何贡献给主成分
# 打印输出 / Print output
print("Feature contributions to first 2 principal components:")
# 打印输出 / Print output
print(f"\nPC1 loadings (importance of each feature):")
# 同时获取索引和值 / Get both index and value
for i, (feature, loading) in enumerate(zip(irisdata['feature_names'], pca.components_[0])):
    # 打印输出 / Print output
    print(f"  {feature}: {loading:.4f}")

# 打印输出 / Print output
print(f"\nPC2 loadings:")
# 同时获取索引和值 / Get both index and value
for i, (feature, loading) in enumerate(zip(irisdata['feature_names'], pca.components_[1])):
    # 打印输出 / Print output
    print(f"  {feature}: {loading:.4f}")
```

## Step 7 — Transform Data and Visualize / 变换数据并可视化

```python
# Transform to PC space and visualize
# 变换到PC空间并可视化
# 用已拟合的模型转换数据 / Transform data with fitted model
X_pca = pca.transform(X)

# 创建画布 / Create figure canvas
plt.figure(figsize=(8, 6))
# 同时获取索引和值 / Get both index and value
for i, name in enumerate(irisdata['target_names']):
    indices = y == i
    # 绘制散点图 / Draw scatter plot
    plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=name, alpha=0.7)
# 设置X轴标签 / Set X-axis label
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
# 设置Y轴标签 / Set Y-axis label
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
# 设置图表标题 / Set chart title
plt.title('Iris Dataset in Principal Component Space')
# 显示图例 / Show legend
plt.legend()
plt.grid(True, alpha=0.3)
# 显示图表 / Display the plot
plt.show()
```

## Step 8 — Train/Test Split / 训练/测试分割

```python
# Split data for classification
# 分割数据用于分类
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Training set size: {X_train.shape[0]}")
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Test set size: {X_test.shape[0]}")
```

## Step 9 — Compare Classifiers / 比较分类器

```python
# Train SVM with all features
# 用所有特征训练SVM
# 支持向量机 / Support Vector Machine
clf_all = SVC(kernel='linear', gamma='auto').fit(X_train, y_train)
y_pred_all = clf_all.predict(X_test)
acc_all = clf_all.score(X_test, y_test)
# 计算F1分数 = 精确率和召回率的调和均值 / F1 = harmonic mean of precision and recall
f1_all = f1_score(y_test, y_pred_all, average='macro')

# 打印输出 / Print output
print("Using all 4 features:")
# 打印输出 / Print output
print(f"  Accuracy: {acc_all:.4f}")
# 打印输出 / Print output
print(f"  F1 Score: {f1_all:.4f}")
```

```python
# Train SVM with only PC1
# 仅用PC1训练SVM
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X_train_pc1 = X_train @ pca.components_[0].reshape(-1, 1)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X_test_pc1 = X_test @ pca.components_[0].reshape(-1, 1)

# 支持向量机 / Support Vector Machine
clf_pc1 = SVC(kernel='linear', gamma='auto').fit(X_train_pc1, y_train)
y_pred_pc1 = clf_pc1.predict(X_test_pc1)
acc_pc1 = clf_pc1.score(X_test_pc1, y_test)
# 计算F1分数 = 精确率和召回率的调和均值 / F1 = harmonic mean of precision and recall
f1_pc1 = f1_score(y_test, y_pred_pc1, average='macro')

# 打印输出 / Print output
print(f"\nUsing only PC1 ({pca.explained_variance_ratio_[0]:.1%} variance):")
# 打印输出 / Print output
print(f"  Accuracy: {acc_pc1:.4f}")
# 打印输出 / Print output
print(f"  F1 Score: {f1_pc1:.4f}")
```

## Step 10 — Summary / 总结

```python
# Print comparison summary
# 打印比较总结
# 打印输出 / Print output
print(f"\nComparison Summary:")
# 打印输出 / Print output
print(f"  All features (4D): Accuracy = {acc_all:.4f}")
# 打印输出 / Print output
print(f"  PC1 only (1D):    Accuracy = {acc_pc1:.4f}")
# 打印输出 / Print output
print(f"  Accuracy loss:     {acc_all - acc_pc1:.4f}")
# 打印输出 / Print output
print(f"\nConclusion: PC1 alone captures enough information for reasonable classification")
# 打印输出 / Print output
print(f"despite explaining only {pca.explained_variance_ratio_[0]:.1%} of total variance")
```

## Learning Notes / 学习笔记

- **Math Essence**: PCA extracts linear combinations of original features that maximize variance. These combinations (principal components) can have higher discriminative power for classification than individual original features, even though they explain less individual variance.
  
  **数学本质**：PCA提取最大化方差的原始特征的线性组合。这些组合可能对分类具有比单个原始特征更高的区分能力。

- **ML Application**: (1) Use PCA for dimensionality reduction before classification to reduce overfitting and training time, (2) The number of components should be chosen based on downstream task requirements, not just variance threshold, (3) For Iris, even 1 component achieves good classification, but 2-3 components are preferred for better generalization and understanding data structure.
  
  **ML应用**：(1) 在分类前使用PCA进行降维以减少过拟合和训练时间，(2) 成分数应基于下游任务要求选择，(3) 对于鸢尾花，甚至1个成分也能实现良好分类。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `PCA` | 主成分分析，降维 | Principal Component Analysis, dimensionality reduction |
| `SVM` | 支持向量机 | Support Vector Machine |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.scatter` | 绘制散点图 | Draw scatter plot |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

➡️ **Next / 下一步**: `../chapter_22/01_download_data.ipynb` — Country comparison using World Bank data

## Complete Code / 完整代码一览

```python
# --- Import Section / 导入部分 ---
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import load_iris
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.decomposition import PCA
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import f1_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# --- Load Data / 加载数据 ---
irisdata = load_iris()
X, y = irisdata['data'], irisdata['target']

# --- PCA Analysis / PCA分析 ---
# 主成分分析：降维，保留最重要的特征 / PCA: reduce dimensions, keep key features
pca = PCA().fit(X)
# 打印输出 / Print output
print("Principal components:")
# 打印输出 / Print output
print(pca.components_)
# 打印输出 / Print output
print("Explained variance:")
# 打印输出 / Print output
print(pca.explained_variance_)

# --- Visualization / 可视化 ---
# 用已拟合的模型转换数据 / Transform data with fitted model
X_pca = pca.transform(X)
# 绘制散点图 / Draw scatter plot
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
# 显示图表 / Display the plot
plt.show()

# --- Classification Comparison / 分类比较 ---
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# 支持向量机 / Support Vector Machine
clf = SVC(kernel='linear', gamma='auto').fit(X_train, y_train)
# 打印输出 / Print output
print(f"Accuracy (all features): {clf.score(X_test, y_test)}")
```

---

### Chapter Summary / 章节总结

# Chapter 21 Summary / 第21章总结：Visualization with PCA

## Theme / 主题

PCA's most immediate and intuitive application is visualization. By reducing to 2D or 3D, we can plot high-dimensional datasets and see structure: clusters, separability, outliers. This chapter applies PCA to real datasets (wine, digits, iris) and uses visualization for exploratory analysis. The goal is to move from theory (PCA math) to practice (plotting real data).

PCA最直接和直观的应用是可视化。通过减少到2D或3D，我们可以绘制高维数据集并看到结构：聚类、可分性、异常值。本章将PCA应用于真实数据集（葡萄酒、数字、鸢尾花）并使用可视化进行探索性分析。目标是从理论（PCA数学）转向实践（绘制真实数据）。

## Evolution / 演化路线

```
01_load_wine.ipynb
    └─ Load wine dataset, apply PCA (加载葡萄酒数据集，应用PCA)
    
02_plot_wine_pca.ipynb
    └─ Plot 2D and 3D PCA projections (绘制2D和3D PCA投影)
    
03_load_digits.ipynb
    └─ Load digit images, reduce to 2D (加载数字图像，减少到2D)
    
04_plot_digits_pca.ipynb
    └─ Visualize digit clusters in PCA space (可视化PCA空间中的数字聚类)
    
05_iris_analysis.ipynb
    └─ Full iris analysis: load → PCA → plot → classify (完整的鸢尾花分析)
```

## Progression Logic / 进度逻辑

Visualization with PCA progresses through **simple → complex → applied**:

1. **Wine dataset**: 
   - Simple structured data (13 features, 3 classes)
   - Clear separation in PCA space
   - Understand what good PCA looks like

2. **Digit dataset**:
   - 28×28 images = 784 features (high-dimensional)
   - PCA to 2D shows digit clusters
   - Demonstrates PCA power on real images

3. **Iris dataset**:
   - Classic ML dataset
   - PCA visualization for classification
   - Combine PCA with other ML (SVM classification)

Each dataset teaches different lessons: simplicity → complexity → practical application.

用PCA进行可视化通过**简单→复杂→应用**进行：

1. **葡萄酒数据集**：
   - 简单结构化数据（13个特征，3个类别）
   - PCA空间中清晰分离
   - 理解好的PCA是什么样的

2. **数字数据集**：
   - 28×28图像= 784个特征（高维）
   - PCA到2D显示数字聚类
   - 演示PCA在真实图像上的能力

3. **鸢尾花数据集**：
   - 经典ML数据集
   - 用于分类的PCA可视化
   - 将PCA与其他ML结合（SVM分类）

每个数据集教导不同的课程：简单→复杂→实际应用。

## ML Relevance / 机器学习相关性

In machine learning:
- **Exploratory data analysis (EDA)**:
  - Apply PCA, plot 2D or 3D
  - Look for clusters, outliers, class separation
  - Guides decisions: are classes separable? linear or nonlinear?
  - Informs choice of algorithm (SVM vs. neural nets)

- **What PCA visualization tells you**:
  - **Well-separated clusters**: Linear classifier should work
  - **Overlapping clusters**: Need nonlinear classifier or more features
  - **Outliers**: Possible data quality issues, consider filtering
  - **One long tail**: First PC captures most variance (good!)
  - **Variance spread evenly**: Many PCs needed (data is high intrinsic dim)

- **Feature engineering via PCA**:
  - Sometimes PCA features beat raw features
  - Decorrelation: PCA outputs are uncorrelated
  - Noise reduction: small eigenvalues are mostly noise
  - Dimensionality: k PCA components instead of n raw features

- **When to use PCA visualization**:
  - High-dimensional data (n > 3): can't plot directly
  - Exploratory analysis: understand data structure
  - Debugging: visualize to find data quality issues
  - Communicating results: 2D plots are easy to explain

- **Limitations**:
  - Linear only: nonlinear structure hidden
  - Unsupervised: ignores labels (could use t-SNE or UMAP for labeled data)
  - Interpretation: PCs are linear combinations (harder to explain than raw features)
  - 2D loss: projecting 1000D to 2D loses 99.8% information

- **Iris + SVM example**:
  - Visualize with PCA to understand separability
  - Train on raw features (13 or 4 for iris)
  - Use PCA visualization to debug model behavior
  - Combine insights: if PCA shows overlap, expect lower accuracy

**Why visualization matters**: A plot is worth a thousand numbers. Understanding data distribution beats abstract statistics.

在机器学习中：
- **探索性数据分析(EDA)**：
  - 应用PCA，绘制2D或3D
  - 查找聚类、异常值、类分离
  - 指导决策：类是否可分？线性或非线性？
  - 通知算法选择（SVM vs.神经网络）

- **PCA可视化告诉您什么**：
  - **良好分离的聚类**：线性分类器应该有效
  - **重叠的聚类**：需要非线性分类器或更多特征
  - **异常值**：可能的数据质量问题，考虑过滤
  - **一条长尾**：第一台PC捕获大部分方差（好！）
  - **方差均匀分布**：需要许多PC（数据是高固有维）

- **通过PCA的特征工程**：
  - 有时PCA特征胜过原始特征
  - 去相关：PCA输出不相关
  - 噪声减少：小特征值主要是噪声
  - 维数：k个PCA分量代替n个原始特征

- **何时使用PCA可视化**：
  - 高维数据（n > 3）：无法直接绘制
  - 探索性分析：理解数据结构
  - 调试：可视化以找到数据质量问题
  - 传达结果：2D绘图易于解释

- **限制**：
  - 仅线性：非线性结构隐藏
  - 无监督：忽略标签（对于标记数据可以使用t-SNE或UMAP）
  - 解释：PC是线性组合（比原始特征更难解释）
  - 2D损失：将1000D投影到2D会丧失99.8%的信息

- **Iris + SVM示例**：
  - 使用PCA可视化以理解可分性
  - 在原始特征上训练（13或4个用于鸢尾花）
  - 使用PCA可视化调试模型行为
  - 组合见解：如果PCA显示重叠，预期准确度较低

**为什么可视化重要**：一张图值一千个数字。理解数据分布胜过抽象统计。

---
