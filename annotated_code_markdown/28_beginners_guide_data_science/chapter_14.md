# 数据科学入门 / Beginner's Guide to Data Science
## Chapter 14

---

### Plot

# 01 — Plot / 01 Plot

**Chapter 14 — File 1 of 14 / 第14章 — 第1个文件（共14个）**

---

## Summary / 总结

This script demonstrates **Load the dataset**.

本脚本演示 **Load the dataset**。

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
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
import seaborn as sns
```

---
## Step 2 — Load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 3 — Calculate skewness

```python
sale_price_skew = Ames['SalePrice'].skew()
year_built_skew = Ames['YearBuilt'].skew()
```

---
## Step 4 — Set the style of seaborn

```python
sns.set(style='whitegrid')
```

---
## Step 5 — Create a figure for 2 subplots (1 row, 2 columns)

```python
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
```

---
## Step 6 — Plot for SalePrice (positively skewed)

```python
sns.histplot(Ames['SalePrice'], kde=True, ax=ax[0], color='skyblue')
ax[0].set_title('Distribution of SalePrice (Positive Skew)', fontsize=16)
ax[0].set_xlabel('SalePrice')
ax[0].set_ylabel('Frequency')
```

---
## Step 7 — Annotate Skewness

```python
ax[0].text(0.5, 0.5, f'Skew: {sale_price_skew:.2f}', transform=ax[0].transAxes,
           horizontalalignment='right', color='black', weight='bold',
           fontsize=14)
```

---
## Step 8 — Plot for YearBuilt (negatively skewed)

```python
sns.histplot(Ames['YearBuilt'], kde=True, ax=ax[1], color='salmon')
ax[1].set_title('Distribution of YearBuilt (Negative Skew)', fontsize=16)
ax[1].set_xlabel('YearBuilt')
ax[1].set_ylabel('Frequency')
```

---
## Step 9 — Annotate Skewness

```python
ax[1].text(0.5, 0.5, f'Skew: {year_built_skew:.2f}', transform=ax[1].transAxes,
           horizontalalignment='right', color='black', weight='bold',
           fontsize=14)

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
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
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
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
# Plot / 01 Plot
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Calculate skewness
sale_price_skew = Ames['SalePrice'].skew()
year_built_skew = Ames['YearBuilt'].skew()

# Set the style of seaborn
sns.set(style='whitegrid')

# Create a figure for 2 subplots (1 row, 2 columns)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot for SalePrice (positively skewed)
sns.histplot(Ames['SalePrice'], kde=True, ax=ax[0], color='skyblue')
ax[0].set_title('Distribution of SalePrice (Positive Skew)', fontsize=16)
ax[0].set_xlabel('SalePrice')
ax[0].set_ylabel('Frequency')

# Annotate Skewness
ax[0].text(0.5, 0.5, f'Skew: {sale_price_skew:.2f}', transform=ax[0].transAxes,
           horizontalalignment='right', color='black', weight='bold',
           fontsize=14)

# Plot for YearBuilt (negatively skewed)
sns.histplot(Ames['YearBuilt'], kde=True, ax=ax[1], color='salmon')
ax[1].set_title('Distribution of YearBuilt (Negative Skew)', fontsize=16)
ax[1].set_xlabel('YearBuilt')
ax[1].set_ylabel('Frequency')

# Annotate Skewness
ax[1].text(0.5, 0.5, f'Skew: {year_built_skew:.2f}', transform=ax[1].transAxes,
           horizontalalignment='right', color='black', weight='bold',
           fontsize=14)

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 2 of 14

---

### Log

# 02 — Log / 02 Log

**Chapter 14 — File 2 of 14 / 第14章 — 第2个文件（共14个）**

---

## Summary / 总结

This script demonstrates **Applying Log Transformation**.

本脚本演示 **Applying Log Transformation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Applying Log Transformation

```python
Ames['Log_SalePrice'] = np.log(Ames['SalePrice'])
# 打印输出 / Print output
print(f"Skewness after Log Transformation: {Ames['Log_SalePrice'].skew():.5f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Applying Log Transformation 是机器学习中的常用技术。  
  *Applying Log Transformation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Log / 02 Log
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Applying Log Transformation
Ames['Log_SalePrice'] = np.log(Ames['SalePrice'])
# 打印输出 / Print output
print(f"Skewness after Log Transformation: {Ames['Log_SalePrice'].skew():.5f}")
```

---

➡️ **Next / 下一步**: File 3 of 14

---

### Sqroot

# 03 — Sqroot / 03 Sqroot

**Chapter 14 — File 3 of 14 / 第14章 — 第3个文件（共14个）**

---

## Summary / 总结

This script demonstrates **Applying Square Root Transformation**.

本脚本演示 **Applying Square Root Transformation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Applying Square Root Transformation

```python
Ames['Sqrt_SalePrice'] = np.sqrt(Ames['SalePrice'])
# 打印输出 / Print output
print(f"Skewness after Square Root Transformation: {Ames['Sqrt_SalePrice'].skew():.5f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Applying Square Root Transformation 是机器学习中的常用技术。  
  *Applying Square Root Transformation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Sqroot / 03 Sqroot
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Applying Square Root Transformation
Ames['Sqrt_SalePrice'] = np.sqrt(Ames['SalePrice'])
# 打印输出 / Print output
print(f"Skewness after Square Root Transformation: {Ames['Sqrt_SalePrice'].skew():.5f}")
```

---

➡️ **Next / 下一步**: File 4 of 14

---

### Boxcox

# 04 — Boxcox / 04 Boxcox

**Chapter 14 — File 4 of 14 / 第14章 — 第4个文件（共14个）**

---

## Summary / 总结

This script demonstrates **Applying Box-Cox Transformation after checking all values are positive**.

本脚本演示 **Applying Box-Cox Transformation after checking all values are positive**。

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
```

---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import scipy.stats

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Applying Box-Cox Transformation after checking all values are positive

```python
if (Ames['SalePrice'] > 0).all():
    Ames['BoxCox_SalePrice'], lmbda = scipy.stats.boxcox(Ames['SalePrice'])
else:
```

---
## Step 3 — Consider alternative transformations or handling strategies

```python
# 打印输出 / Print output
print("Not all SalePrice values are positive.")
    # 打印输出 / Print output
    print("Consider using Yeo-Johnson or handling negative values.")
# 打印输出 / Print output
print(f"Skewness after Box-Cox Transformation: {Ames['BoxCox_SalePrice'].skew():.5f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Applying Box-Cox Transformation after checking all values are positive 是机器学习中的常用技术。  
  *Applying Box-Cox Transformation after checking all values are positive is a common technique in machine learning.*

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
# Boxcox / 04 Boxcox
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import scipy.stats

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Applying Box-Cox Transformation after checking all values are positive
if (Ames['SalePrice'] > 0).all():
    Ames['BoxCox_SalePrice'], lmbda = scipy.stats.boxcox(Ames['SalePrice'])
else:
    # Consider alternative transformations or handling strategies
    # 打印输出 / Print output
    print("Not all SalePrice values are positive.")
    # 打印输出 / Print output
    print("Consider using Yeo-Johnson or handling negative values.")
# 打印输出 / Print output
print(f"Skewness after Box-Cox Transformation: {Ames['BoxCox_SalePrice'].skew():.5f}")
```

---

➡️ **Next / 下一步**: File 5 of 14

---

### Yeo

# 05 — Yeo / 05 Yeo

**Chapter 14 — File 5 of 14 / 第14章 — 第5个文件（共14个）**

---

## Summary / 总结

This script demonstrates **Applying Yeo-Johnson Transformation**.

本脚本演示 **Applying Yeo-Johnson Transformation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import scipy.stats

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Applying Yeo-Johnson Transformation

```python
Ames['YeoJohnson_SalePrice'], _ = scipy.stats.yeojohnson(Ames['SalePrice'])
# 打印输出 / Print output
print("Skewness after Yeo-Johnson Transformation: "
      f"{Ames['YeoJohnson_SalePrice'].skew():.5f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Applying Yeo-Johnson Transformation 是机器学习中的常用技术。  
  *Applying Yeo-Johnson Transformation is a common technique in machine learning.*

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
# Yeo / 05 Yeo
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import scipy.stats

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Applying Yeo-Johnson Transformation
Ames['YeoJohnson_SalePrice'], _ = scipy.stats.yeojohnson(Ames['SalePrice'])
# 打印输出 / Print output
print("Skewness after Yeo-Johnson Transformation: "
      f"{Ames['YeoJohnson_SalePrice'].skew():.5f}")
```

---

➡️ **Next / 下一步**: File 6 of 14

---

### Quantile

# 06 — Quantile / 06 Quantile

**Chapter 14 — File 6 of 14 / 第14章 — 第6个文件（共14个）**

---

## Summary / 总结

This script demonstrates **Applying Quantile Transformation to follow a normal distribution**.

本脚本演示 **Applying Quantile Transformation to follow a normal distribution**。

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
```

---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import QuantileTransformer

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Applying Quantile Transformation to follow a normal distribution

```python
quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
Ames['Quantile_SalePrice'] = \
    # 转换为NumPy数组 / Convert to NumPy array
    quantile_transformer.fit_transform(Ames['SalePrice'].values.reshape(-1, 1)).flatten()
# 打印输出 / Print output
print(f"Skewness after Quantile Transformation: {Ames['Quantile_SalePrice'].skew():.5f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Applying Quantile Transformation to follow a normal distribution 是机器学习中的常用技术。  
  *Applying Quantile Transformation to follow a normal distribution is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Quantile / 06 Quantile
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import QuantileTransformer

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Applying Quantile Transformation to follow a normal distribution
quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
Ames['Quantile_SalePrice'] = \
    # 转换为NumPy数组 / Convert to NumPy array
    quantile_transformer.fit_transform(Ames['SalePrice'].values.reshape(-1, 1)).flatten()
# 打印输出 / Print output
print(f"Skewness after Quantile Transformation: {Ames['Quantile_SalePrice'].skew():.5f}")
```

---

➡️ **Next / 下一步**: File 7 of 14

---

### Plot

# 07 — Plot / 07 Plot

**Chapter 14 — File 7 of 14 / 第14章 — 第7个文件（共14个）**

---

## Summary / 总结

This script demonstrates **Plotting the distributions**.

本脚本演示 **Plotting the distributions**。

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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
import scipy.stats
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
import seaborn as sns
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import QuantileTransformer

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
Ames['Log_SalePrice'] = np.log(Ames['SalePrice'])
Ames['Sqrt_SalePrice'] = np.sqrt(Ames['SalePrice'])
Ames['BoxCox_SalePrice'], _ = scipy.stats.boxcox(Ames['SalePrice'])
Ames['YeoJohnson_SalePrice'], _ = scipy.stats.yeojohnson(Ames['SalePrice'])
quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
Ames['Quantile_SalePrice'] = \
    # 转换为NumPy数组 / Convert to NumPy array
    quantile_transformer.fit_transform(Ames['SalePrice'].values.reshape(-1, 1)).flatten()
```

---
## Step 2 — Plotting the distributions

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
```

---
## Step 3 — Flatten the axes array for easier indexing

```python
# 展平为一维数组 / Flatten to 1D array
axes = axes.flatten()
```

---
## Step 4 — Hide unused subplot axes

```python
for ax in axes[6:]:
    ax.axis('off')
```

---
## Step 5 — Original SalePrice Distribution

```python
sns.histplot(Ames['SalePrice'], kde=True, bins=30, color='skyblue', ax=axes[0])
axes[0].set_title('Original SalePrice Distribution (Skew: 1.76)')
axes[0].set_xlabel('SalePrice')
axes[0].set_ylabel('Frequency')
```

---
## Step 6 — Log Transformed SalePrice

```python
sns.histplot(Ames['Log_SalePrice'], kde=True, bins=30, color='blue', ax=axes[1])
axes[1].set_title('Log Transformed SalePrice (Skew: 0.04172)')
axes[1].set_xlabel('Log of SalePrice')
axes[1].set_ylabel('Frequency')
```

---
## Step 7 — Square Root Transformed SalePrice

```python
sns.histplot(Ames['Sqrt_SalePrice'], kde=True, bins=30, color='orange', ax=axes[2])
axes[2].set_title('Square Root Transformed (Skew: 0.90148)')
axes[2].set_xlabel('Square Root of SalePrice')
axes[2].set_ylabel('Frequency')
```

---
## Step 8 — Box-Cox Transformed SalePrice

```python
sns.histplot(Ames['BoxCox_SalePrice'], kde=True, bins=30, color='red', ax=axes[3])
axes[3].set_title('Box-Cox Transformed SalePrice (Skew: -0.00436)')
axes[3].set_xlabel('Box-Cox of SalePrice')
axes[3].set_ylabel('Frequency')
```

---
## Step 9 — Yeo-Johnson Transformed SalePrice

```python
sns.histplot(Ames['YeoJohnson_SalePrice'], kde=True, bins=30, color='purple', ax=axes[4])
axes[4].set_title('Yeo-Johnson Transformed (Skew: -0.00437)')
axes[4].set_xlabel('Yeo-Johnson of SalePrice')
axes[4].set_ylabel('Frequency')
```

---
## Step 10 — Quantile Transformed SalePrice (Normal Distribution)

```python
sns.histplot(Ames['Quantile_SalePrice'], kde=True, bins=30, color='green', ax=axes[5])
axes[5].set_title('Quantile Transformed (Normal Distn, Skew: 0.00286)')
axes[5].set_xlabel('Quantile Transformed SalePrice')
axes[5].set_ylabel('Frequency')

plt.tight_layout(pad=4.0)
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Plotting the distributions 是机器学习中的常用技术。  
  *Plotting the distributions is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot / 07 Plot
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
import scipy.stats
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
import seaborn as sns
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import QuantileTransformer

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
Ames['Log_SalePrice'] = np.log(Ames['SalePrice'])
Ames['Sqrt_SalePrice'] = np.sqrt(Ames['SalePrice'])
Ames['BoxCox_SalePrice'], _ = scipy.stats.boxcox(Ames['SalePrice'])
Ames['YeoJohnson_SalePrice'], _ = scipy.stats.yeojohnson(Ames['SalePrice'])
quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
Ames['Quantile_SalePrice'] = \
    # 转换为NumPy数组 / Convert to NumPy array
    quantile_transformer.fit_transform(Ames['SalePrice'].values.reshape(-1, 1)).flatten()

# Plotting the distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Flatten the axes array for easier indexing
# 展平为一维数组 / Flatten to 1D array
axes = axes.flatten()

# Hide unused subplot axes
for ax in axes[6:]:
    ax.axis('off')

# Original SalePrice Distribution
sns.histplot(Ames['SalePrice'], kde=True, bins=30, color='skyblue', ax=axes[0])
axes[0].set_title('Original SalePrice Distribution (Skew: 1.76)')
axes[0].set_xlabel('SalePrice')
axes[0].set_ylabel('Frequency')

# Log Transformed SalePrice
sns.histplot(Ames['Log_SalePrice'], kde=True, bins=30, color='blue', ax=axes[1])
axes[1].set_title('Log Transformed SalePrice (Skew: 0.04172)')
axes[1].set_xlabel('Log of SalePrice')
axes[1].set_ylabel('Frequency')

# Square Root Transformed SalePrice
sns.histplot(Ames['Sqrt_SalePrice'], kde=True, bins=30, color='orange', ax=axes[2])
axes[2].set_title('Square Root Transformed (Skew: 0.90148)')
axes[2].set_xlabel('Square Root of SalePrice')
axes[2].set_ylabel('Frequency')

# Box-Cox Transformed SalePrice
sns.histplot(Ames['BoxCox_SalePrice'], kde=True, bins=30, color='red', ax=axes[3])
axes[3].set_title('Box-Cox Transformed SalePrice (Skew: -0.00436)')
axes[3].set_xlabel('Box-Cox of SalePrice')
axes[3].set_ylabel('Frequency')

# Yeo-Johnson Transformed SalePrice
sns.histplot(Ames['YeoJohnson_SalePrice'], kde=True, bins=30, color='purple', ax=axes[4])
axes[4].set_title('Yeo-Johnson Transformed (Skew: -0.00437)')
axes[4].set_xlabel('Yeo-Johnson of SalePrice')
axes[4].set_ylabel('Frequency')

# Quantile Transformed SalePrice (Normal Distribution)
sns.histplot(Ames['Quantile_SalePrice'], kde=True, bins=30, color='green', ax=axes[5])
axes[5].set_title('Quantile Transformed (Normal Distn, Skew: 0.00286)')
axes[5].set_xlabel('Quantile Transformed SalePrice')
axes[5].set_ylabel('Frequency')

plt.tight_layout(pad=4.0)
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 8 of 14

---

### Squared

# 08 — Squared / 08 Squared

**Chapter 14 — File 8 of 14 / 第14章 — 第8个文件（共14个）**

---

## Summary / 总结

This script demonstrates **Applying Squared Transformation**.

本脚本演示 **Applying Squared Transformation**。

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
## Step 2 — Applying Squared Transformation

```python
Ames['Squared_YearBuilt'] = Ames['YearBuilt'] ** 2
# 打印输出 / Print output
print(f"Skewness after Squared Transformation: {Ames['Squared_YearBuilt'].skew():.5f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Applying Squared Transformation 是机器学习中的常用技术。  
  *Applying Squared Transformation is a common technique in machine learning.*

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
# Squared / 08 Squared
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Applying Squared Transformation
Ames['Squared_YearBuilt'] = Ames['YearBuilt'] ** 2
# 打印输出 / Print output
print(f"Skewness after Squared Transformation: {Ames['Squared_YearBuilt'].skew():.5f}")
```

---

➡️ **Next / 下一步**: File 9 of 14

---

### Cubed

# 09 — Cubed / 09 Cubed

**Chapter 14 — File 9 of 14 / 第14章 — 第9个文件（共14个）**

---

## Summary / 总结

This script demonstrates **Applying Cubed Transformation**.

本脚本演示 **Applying Cubed Transformation**。

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
## Step 2 — Applying Cubed Transformation

```python
Ames['Cubed_YearBuilt'] = Ames['YearBuilt'] ** 3
# 打印输出 / Print output
print(f"Skewness after Cubed Transformation: {Ames['Cubed_YearBuilt'].skew():.5f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Applying Cubed Transformation 是机器学习中的常用技术。  
  *Applying Cubed Transformation is a common technique in machine learning.*

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
# Cubed / 09 Cubed
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Applying Cubed Transformation
Ames['Cubed_YearBuilt'] = Ames['YearBuilt'] ** 3
# 打印输出 / Print output
print(f"Skewness after Cubed Transformation: {Ames['Cubed_YearBuilt'].skew():.5f}")
```

---

➡️ **Next / 下一步**: File 10 of 14

---

### Boxcox

# 10 — Boxcox / 10 Boxcox

**Chapter 14 — File 10 of 14 / 第14章 — 第10个文件（共14个）**

---

## Summary / 总结

This script demonstrates **Applying Box-Cox Transformation after checking all values are positive**.

本脚本演示 **Applying Box-Cox Transformation after checking all values are positive**。

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
```

---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import scipy.stats

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Applying Box-Cox Transformation after checking all values are positive

```python
if (Ames['YearBuilt'] > 0).all():
    Ames['BoxCox_YearBuilt'], _ = scipy.stats.boxcox(Ames['YearBuilt'])
else:
```

---
## Step 3 — Consider alternative transformations or handling strategies

```python
# 打印输出 / Print output
print("Not all YearBuilt values are positive.")
    # 打印输出 / Print output
    print("Consider using Yeo-Johnson or handling negative values.")
# 打印输出 / Print output
print(f"Skewness after Box-Cox Transformation: {Ames['BoxCox_YearBuilt'].skew():.5f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Applying Box-Cox Transformation after checking all values are positive 是机器学习中的常用技术。  
  *Applying Box-Cox Transformation after checking all values are positive is a common technique in machine learning.*

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
# Boxcox / 10 Boxcox
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import scipy.stats

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Applying Box-Cox Transformation after checking all values are positive
if (Ames['YearBuilt'] > 0).all():
    Ames['BoxCox_YearBuilt'], _ = scipy.stats.boxcox(Ames['YearBuilt'])
else:
    # Consider alternative transformations or handling strategies
    # 打印输出 / Print output
    print("Not all YearBuilt values are positive.")
    # 打印输出 / Print output
    print("Consider using Yeo-Johnson or handling negative values.")
# 打印输出 / Print output
print(f"Skewness after Box-Cox Transformation: {Ames['BoxCox_YearBuilt'].skew():.5f}")
```

---

➡️ **Next / 下一步**: File 11 of 14

---

### Yeo

# 11 — Yeo / 11 Yeo

**Chapter 14 — File 11 of 14 / 第14章 — 第11个文件（共14个）**

---

## Summary / 总结

This script demonstrates **Applying Yeo-Johnson Transformation**.

本脚本演示 **Applying Yeo-Johnson Transformation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import scipy.stats

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Applying Yeo-Johnson Transformation

```python
Ames['YeoJohnson_YearBuilt'], _ = scipy.stats.yeojohnson(Ames['YearBuilt'])
# 打印输出 / Print output
print("Skewness after Yeo-Johnson Transformation: "
      f"{Ames['YeoJohnson_YearBuilt'].skew():.5f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Applying Yeo-Johnson Transformation 是机器学习中的常用技术。  
  *Applying Yeo-Johnson Transformation is a common technique in machine learning.*

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
# Yeo / 11 Yeo
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import scipy.stats

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Applying Yeo-Johnson Transformation
Ames['YeoJohnson_YearBuilt'], _ = scipy.stats.yeojohnson(Ames['YearBuilt'])
# 打印输出 / Print output
print("Skewness after Yeo-Johnson Transformation: "
      f"{Ames['YeoJohnson_YearBuilt'].skew():.5f}")
```

---

➡️ **Next / 下一步**: File 12 of 14

---

### Quantile

# 12 — Quantile / 12 Quantile

**Chapter 14 — File 12 of 14 / 第14章 — 第12个文件（共14个）**

---

## Summary / 总结

This script demonstrates **Applying Quantile Transformation to follow a normal distribution**.

本脚本演示 **Applying Quantile Transformation to follow a normal distribution**。

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
```

---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import QuantileTransformer

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Applying Quantile Transformation to follow a normal distribution

```python
quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
Ames['Quantile_YearBuilt'] = \
    # 转换为NumPy数组 / Convert to NumPy array
    quantile_transformer.fit_transform(Ames['YearBuilt'].values.reshape(-1, 1)).flatten()
# 打印输出 / Print output
print(f"Skewness after Quantile Transformation: {Ames['Quantile_YearBuilt'].skew():.5f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Applying Quantile Transformation to follow a normal distribution 是机器学习中的常用技术。  
  *Applying Quantile Transformation to follow a normal distribution is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Quantile / 12 Quantile
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import QuantileTransformer

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Applying Quantile Transformation to follow a normal distribution
quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
Ames['Quantile_YearBuilt'] = \
    # 转换为NumPy数组 / Convert to NumPy array
    quantile_transformer.fit_transform(Ames['YearBuilt'].values.reshape(-1, 1)).flatten()
# 打印输出 / Print output
print(f"Skewness after Quantile Transformation: {Ames['Quantile_YearBuilt'].skew():.5f}")
```

---

➡️ **Next / 下一步**: File 13 of 14

---

### Plot



---

### Kstest

# 14 — Kstest / 14 Kstest

**Chapter 14 — File 14 of 14 / 第14章 — 第14个文件（共14个）**

---

## Summary / 总结

This script demonstrates **Run the Kolmogorov-Smirnov tests for the 10 cases**.

本脚本演示 **Run the Kolmogorov-Smirnov tests for the 10 cases**。

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
```

---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
import scipy.stats
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import QuantileTransformer

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
Ames['Log_SalePrice'] = np.log(Ames['SalePrice'])
Ames['Sqrt_SalePrice'] = np.sqrt(Ames['SalePrice'])
Ames['BoxCox_SalePrice'], _ = scipy.stats.boxcox(Ames['SalePrice'])
Ames['YeoJohnson_SalePrice'], _ = scipy.stats.yeojohnson(Ames['SalePrice'])
quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
Ames['Quantile_SalePrice'] = \
    # 转换为NumPy数组 / Convert to NumPy array
    quantile_transformer.fit_transform(Ames['SalePrice'].values.reshape(-1, 1)).flatten()
Ames['Squared_YearBuilt'] = Ames['YearBuilt'] ** 2
Ames['Cubed_YearBuilt'] = Ames['YearBuilt'] ** 3
Ames['BoxCox_YearBuilt'], _ = scipy.stats.boxcox(Ames['YearBuilt'])
Ames['YeoJohnson_YearBuilt'], _ = scipy.stats.yeojohnson(Ames['YearBuilt'])
Ames['Quantile_YearBuilt'] = \
    # 转换为NumPy数组 / Convert to NumPy array
    quantile_transformer.fit_transform(Ames['YearBuilt'].values.reshape(-1, 1)).flatten()
```

---
## Step 2 — Run the Kolmogorov-Smirnov tests for the 10 cases

```python
transformations = ["Log_SalePrice", "Sqrt_SalePrice", "BoxCox_SalePrice",
                    "YeoJohnson_SalePrice", "Quantile_SalePrice",
                    "Squared_YearBuilt", "Cubed_YearBuilt", "BoxCox_YearBuilt",
                    "YeoJohnson_YearBuilt", "Quantile_YearBuilt"]
```

---
## Step 3 — Standardizing the transformations before performing KS test

```python
ks_test_results = {}
for transformation in transformations:
    standardized_data = \
        (Ames[transformation] - Ames[transformation].mean()) / Ames[transformation].std()
    ks_stat, ks_p_value = scipy.stats.kstest(standardized_data, 'norm')
    ks_test_results[transformation] = (ks_stat, ks_p_value)
```

---
## Step 4 — Convert results to DataFrame for easier comparison

```python
ks_test_results_df = pd.DataFrame.from_dict(ks_test_results, orient='index',
                                            columns=['KS Statistic', 'P-Value'])
# 打印输出 / Print output
print(ks_test_results_df.round(5))
```

---
## Learning Notes / 学习笔记

- **概念**: Run the Kolmogorov-Smirnov tests for the 10 cases 是机器学习中的常用技术。  
  *Run the Kolmogorov-Smirnov tests for the 10 cases is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Kstest / 14 Kstest
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
import scipy.stats
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import QuantileTransformer

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
Ames['Log_SalePrice'] = np.log(Ames['SalePrice'])
Ames['Sqrt_SalePrice'] = np.sqrt(Ames['SalePrice'])
Ames['BoxCox_SalePrice'], _ = scipy.stats.boxcox(Ames['SalePrice'])
Ames['YeoJohnson_SalePrice'], _ = scipy.stats.yeojohnson(Ames['SalePrice'])
quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
Ames['Quantile_SalePrice'] = \
    # 转换为NumPy数组 / Convert to NumPy array
    quantile_transformer.fit_transform(Ames['SalePrice'].values.reshape(-1, 1)).flatten()
Ames['Squared_YearBuilt'] = Ames['YearBuilt'] ** 2
Ames['Cubed_YearBuilt'] = Ames['YearBuilt'] ** 3
Ames['BoxCox_YearBuilt'], _ = scipy.stats.boxcox(Ames['YearBuilt'])
Ames['YeoJohnson_YearBuilt'], _ = scipy.stats.yeojohnson(Ames['YearBuilt'])
Ames['Quantile_YearBuilt'] = \
    # 转换为NumPy数组 / Convert to NumPy array
    quantile_transformer.fit_transform(Ames['YearBuilt'].values.reshape(-1, 1)).flatten()

# Run the Kolmogorov-Smirnov tests for the 10 cases
transformations = ["Log_SalePrice", "Sqrt_SalePrice", "BoxCox_SalePrice",
                    "YeoJohnson_SalePrice", "Quantile_SalePrice",
                    "Squared_YearBuilt", "Cubed_YearBuilt", "BoxCox_YearBuilt",
                    "YeoJohnson_YearBuilt", "Quantile_YearBuilt"]

# Standardizing the transformations before performing KS test
ks_test_results = {}
for transformation in transformations:
    standardized_data = \
        (Ames[transformation] - Ames[transformation].mean()) / Ames[transformation].std()
    ks_stat, ks_p_value = scipy.stats.kstest(standardized_data, 'norm')
    ks_test_results[transformation] = (ks_stat, ks_p_value)

# Convert results to DataFrame for easier comparison
ks_test_results_df = pd.DataFrame.from_dict(ks_test_results, orient='index',
                                            columns=['KS Statistic', 'P-Value'])
# 打印输出 / Print output
print(ks_test_results_df.round(5))
```

---

### Chapter Summary / 章节总结

# Chapter 14 Summary / 第14章总结

## Theme / 主题: Chapter 14 / Chapter 14

This chapter contains **14 code files** demonstrating chapter 14.

本章包含 **14 个代码文件**，演示Chapter 14。

---
## Evolution / 演化路线

  1. `01_plot.ipynb` — Plot
  2. `02_log.ipynb` — Log
  3. `03_sqroot.ipynb` — Sqroot
  4. `04_boxcox.ipynb` — Boxcox
  5. `05_yeo.ipynb` — Yeo
  6. `06_quantile.ipynb` — Quantile
  7. `07_plot.ipynb` — Plot
  8. `08_squared.ipynb` — Squared
  9. `09_cubed.ipynb` — Cubed
  10. `10_boxcox.ipynb` — Boxcox
  11. `11_yeo.ipynb` — Yeo
  12. `12_quantile.ipynb` — Quantile
  13. `13_plot.ipynb` — Plot
  14. `14_kstest.ipynb` — Kstest

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 14) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 14）是机器学习流水线中的基础构建块。

---
