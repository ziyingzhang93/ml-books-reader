# 统计方法与机器学习 / Statistical Methods for Machine Learning
## Appendix 02

---

### Versions

# Appendix 02 — Library Versions / 库版本

**Appendix 02 — File 1 of 1**

## Summary / 摘要

This notebook displays the version numbers of key scientific computing and machine learning libraries used throughout the statistical methods course. Version tracking is essential for reproducibility, ensuring that code written with specific library versions can be verified and run reliably in different environments. Each library plays a critical role in the data science pipeline.

本笔记本显示整个统计方法课程中使用的关键科学计算和机器学习库的版本号。版本跟踪对于可重复性至关重要，确保用特定库版本编写的代码可以在不同环境中可靠地验证和运行。每个库在数据科学管道中都发挥关键作用。

### Libraries Checked / 检查的库

- **SciPy**: Statistical computing and scientific computation
- **NumPy**: Numerical computing and array operations
- **Matplotlib**: Data visualization and plotting
- **Pandas**: Data manipulation and analysis
- **Statsmodels**: Statistical modeling and hypothesis testing
- **Scikit-learn**: Machine learning algorithms and tools

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import and Display SciPy Version / 导入并显示SciPy版本

```python
# SciPy: Statistical computing and special functions
# 用于统计计算和特殊函数
import scipy
# 打印输出 / Print output
print('scipy: %s' % scipy.__version__)
```

## Step 2 — Import and Display NumPy Version / 导入并显示NumPy版本

```python
# NumPy: Fundamental package for numerical computing
# 用于数值计算的基础包
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
# 打印输出 / Print output
print('numpy: %s' % numpy.__version__)
```

## Step 3 — Import and Display Matplotlib Version / 导入并显示Matplotlib版本

```python
# Matplotlib: Visualization library for creating plots
# 用于创建图的可视化库
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib
# 打印输出 / Print output
print('matplotlib: %s' % matplotlib.__version__)
```

## Step 4 — Import and Display Pandas Version / 导入并显示Pandas版本

```python
# Pandas: Data manipulation and analysis library
# 用于数据操作和分析的库
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas
# 打印输出 / Print output
print('pandas: %s' % pandas.__version__)
```

## Step 5 — Import and Display Statsmodels Version / 导入并显示Statsmodels版本

```python
# Statsmodels: Statistical models and hypothesis testing
# 用于统计模型和假设检验
import statsmodels
# 打印输出 / Print output
print('statsmodels: %s' % statsmodels.__version__)
```

## Step 6 — Import and Display Scikit-learn Version / 导入并显示Scikit-learn版本

```python
# Scikit-learn: Machine learning library
# 用于机器学习的库
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
import sklearn
# 打印输出 / Print output
print('sklearn: %s' % sklearn.__version__)
```

## Learning Notes / 学习笔记

- **Statistical Concept**: Reproducibility and environment consistency are fundamental to scientific computing. Version tracking allows researchers and practitioners to understand which algorithm implementations, bug fixes, and feature updates were present when code was written. Major version changes often introduce breaking changes in APIs, while minor and patch versions typically maintain backward compatibility.

  **统计概念**: 可重现性和环境一致性是科学计算的基础。版本跟踪允许研究人员和实践者了解代码编写时存在哪些算法实现、bug修复和特征更新。主版本更改通常在API中引入破坏性更改，而次版本和补丁版本通常保持向后兼容性。

- **ML Application**: In production ML systems and research projects, maintaining a requirements.txt or environment.yml file with pinned library versions is critical for reproducibility and deployment consistency. Different versions of libraries like scikit-learn may produce different results due to algorithm updates, random seed changes, or dependency adjustments. Version management is essential for code review, peer verification, and long-term model maintenance.

  **ML应用**: 在生产ML系统和研究项目中，维护带有固定库版本的requirements.txt或environment.yml文件对于可重现性和部署一致性至关重要。不同版本的库(如scikit-learn)可能由于算法更新、随机种子更改或依赖调整而产生不同的结果。版本管理对于代码审查、同行验证和长期模型维护至关重要。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |

➡️ **Next**: This is the final notebook in the course. Congratulations on completing "Statistical Methods for Machine Learning"!

## Complete Code / 完整代码一览

```python
import scipy
# 打印输出 / Print output
print('scipy: %s' % scipy.__version__)

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
# 打印输出 / Print output
print('numpy: %s' % numpy.__version__)

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib
# 打印输出 / Print output
print('matplotlib: %s' % matplotlib.__version__)

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas
# 打印输出 / Print output
print('pandas: %s' % pandas.__version__)

import statsmodels
# 打印输出 / Print output
print('statsmodels: %s' % statsmodels.__version__)

# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
import sklearn
# 打印输出 / Print output
print('sklearn: %s' % sklearn.__version__)
```

---

### Chapter Summary / 章节总结

# Appendix 2: Library Versions
# 附录2：库版本

## Theme | 主题
Environment reproducibility: document version compatibility for all statistical libraries.
环境再现性：为所有统计库记录版本兼容性。

## Evolution Roadmap | 演变路线图
```
Python Version (e.g., 3.8, 3.9, 3.10)
└─ Core Libraries
   ├─ NumPy (numerical computing)
   ├─ SciPy (scientific computing)
   ├─ Pandas (data manipulation)
   ├─ Matplotlib (visualization)
   ├─ Seaborn (statistical visualization)
   └─ Scikit-Learn (machine learning)
      └─ Statsmodels (statistical models)
```

## Progression Logic | 进度逻辑

### Stage 1: Python Version (Python版本)
**English:** Core language version. Different versions have different syntax and performance. Example: 3.8 vs. 3.10+ (walrus operator :=).
**中文:** 核心语言版本。不同版本有不同的语法和性能。例如：3.8 vs. 3.10+(海象运算符:=)。

### Stage 2: NumPy (数字计算)
**English:** Foundation for all numerical Python. Handles arrays, linear algebra, random number generation. Critical for reproducibility (random seed behavior varies by version).
**中文:** 所有数字Python的基础。处理数组、线性代数、随机数生成。对再现性至关重要(随机种子行为因版本而异)。

### Stage 3: SciPy (科学计算)
**English:** Statistical distributions, optimization, integration. Contains scipy.stats (PDFs, CDFs, PPFs, hypothesis tests).
**中文:** 统计分布、优化、积分。包含scipy.stats(PDF、CDF、PPF、假设检验)。

### Stage 4: Pandas (数据操作)
**English:** DataFrames and Series for data cleaning, grouping, and transformation. Essential for real-world data.
**中文:** 用于数据清洁、分组和转换的DataFrame和Series。对真实数据必不可少。

### Stage 5: Matplotlib & Seaborn (可视化)
**English:** Matplotlib: low-level plotting. Seaborn: high-level statistical graphics (built on Matplotlib). Version compatibility affects plot appearance.
**中文:** Matplotlib：低级绘图。Seaborn：高级统计图形(构建在Matplotlib上)。版本兼容性影响图形外观。

### Stage 6: Scikit-Learn & Statsmodels (机器学习与统计)
**English:** Scikit-learn: ML algorithms, model selection, preprocessing. Statsmodels: detailed statistical inference, hypothesis testing, regression diagnostics.
**中文:** Scikit-learn：ML算法、模型选择、预处理。Statsmodels：详细的统计推断、假设检验、回归诊断。

## ML Relevance | ML相关性

1. **Reproducibility (再现性)**: Document versions to ensure code runs identically across systems and over time.
2. **Compatibility (兼容性)**: Major version changes (e.g., NumPy 1 → 2) can break code; minor versions often add features without breaking.
3. **Performance (性能)**: Newer versions often include optimizations. Stay updated but test for compatibility.
4. **Bug Fixes (漏洞修复)**: Libraries evolve; earlier versions may have statistical or numerical bugs. Upgrade cautiously but regularly.


---
