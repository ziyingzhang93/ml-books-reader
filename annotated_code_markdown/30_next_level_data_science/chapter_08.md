# 进阶数据科学 / Next Level Data Science
## Chapter 08

---

### Cv

# 01 — Cv / 01 Cv

**Chapter 08 — File 1 of 4 / 第08章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Import necessary libraries**.

本脚本演示 **Import necessary libraries**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
```

---
## Step 1 — Import necessary libraries

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
```

---
## Step 2 — Prepare data and setup for linear regression

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")
y = Ames["SalePrice"]
linear_model = LinearRegression()
```

---
## Step 3 — Perform 5-fold cross-validation without Pipeline

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
cv_score = cross_val_score(linear_model, Ames[["OverallQual"]], y).mean()
# 打印输出 / Print output
print("Example Without Pipeline, Mean CV R^2 score for 'OverallQual': {:.3f}"
      .format(cv_score))
```

---
## Step 4 — Perform 5-fold cross-validation WITH Pipeline

```python
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline([("regressor", linear_model)])
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
pipeline_score = cross_val_score(pipeline, Ames[["OverallQual"]], y, cv=5).mean()
# 打印输出 / Print output
print("Example With Pipeline, Mean CV R^2 for 'OverallQual': {:.3f}"
      .format(pipeline_score))
```

---
## Learning Notes / 学习笔记

- **概念**: Import necessary libraries 是机器学习中的常用技术。  
  *Import necessary libraries is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cv / 01 Cv
# Complete Code / 完整代码
# ===============================

# Import necessary libraries
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline

# Prepare data and setup for linear regression
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")
y = Ames["SalePrice"]
linear_model = LinearRegression()

# Perform 5-fold cross-validation without Pipeline
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
cv_score = cross_val_score(linear_model, Ames[["OverallQual"]], y).mean()
# 打印输出 / Print output
print("Example Without Pipeline, Mean CV R^2 score for 'OverallQual': {:.3f}"
      .format(cv_score))

# Perform 5-fold cross-validation WITH Pipeline
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline([("regressor", linear_model)])
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
pipeline_score = cross_val_score(pipeline, Ames[["OverallQual"]], y, cv=5).mean()
# 打印输出 / Print output
print("Example With Pipeline, Mean CV R^2 for 'OverallQual': {:.3f}"
      .format(pipeline_score))
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Synthetic

# 02 — Synthetic / 02 Synthetic

**Chapter 08 — File 2 of 4 / 第08章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Import necessary libraries**.

本脚本演示 **Import necessary libraries**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
```

---
## Step 1 — Import necessary libraries

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import FunctionTransformer
```

---
## Step 2 — Prepare data and setup for linear regression

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")
y = Ames["SalePrice"]
linear_model = LinearRegression()
```

---
## Step 3 — Perform 5-fold cross-validation without Pipeline

```python
Ames["OWA"] = Ames["OverallQual"] * Ames["GrLivArea"]
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
cv_score_2 = cross_val_score(linear_model, Ames[["OWA"]], y).mean()
# 打印输出 / Print output
print("Example Without Pipeline, Mean CV R^2 score for 'Quality Weighted Area':"
      "{:.3f}".format(cv_score_2))
```

---
## Step 4 — WITH Pipeline
Define the transformation function for "QualityArea"

```python
def create_quality_area(X):
    X["QualityArea"] = X["OverallQual"] * X["GrLivArea"]
    # 转换为NumPy数组 / Convert to NumPy array
    return X[["QualityArea"]].values
```

---
## Step 5 — Setup the FunctionTransformer using the function

```python
quality_area_transformer = FunctionTransformer(create_quality_area)
```

---
## Step 6 — Pipeline using the engineered feature "QualityArea"

```python
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline_2 = Pipeline([
    ("quality_area_transform", quality_area_transformer),
    ("regressor", linear_model)
])
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
pipeline_score_2 = cross_val_score(pipeline_2, Ames[["OverallQual", "GrLivArea"]], y,
                                   cv=5).mean()
```

---
## Step 7 — Output the mean CV scores rounded to four decimal places

```python
# 打印输出 / Print output
print("Example With Pipeline, Mean CV R^2 score for 'Quality Weighted Area': "
      "{:.3f}".format(pipeline_score_2))
```

---
## Learning Notes / 学习笔记

- **概念**: Import necessary libraries 是机器学习中的常用技术。  
  *Import necessary libraries is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Synthetic / 02 Synthetic
# Complete Code / 完整代码
# ===============================

# Import necessary libraries
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import FunctionTransformer

# Prepare data and setup for linear regression
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")
y = Ames["SalePrice"]
linear_model = LinearRegression()

# Perform 5-fold cross-validation without Pipeline
Ames["OWA"] = Ames["OverallQual"] * Ames["GrLivArea"]
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
cv_score_2 = cross_val_score(linear_model, Ames[["OWA"]], y).mean()
# 打印输出 / Print output
print("Example Without Pipeline, Mean CV R^2 score for 'Quality Weighted Area':"
      "{:.3f}".format(cv_score_2))

# WITH Pipeline
# Define the transformation function for "QualityArea"
def create_quality_area(X):
    X["QualityArea"] = X["OverallQual"] * X["GrLivArea"]
    # 转换为NumPy数组 / Convert to NumPy array
    return X[["QualityArea"]].values

# Setup the FunctionTransformer using the function
quality_area_transformer = FunctionTransformer(create_quality_area)

# Pipeline using the engineered feature "QualityArea"
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline_2 = Pipeline([
    ("quality_area_transform", quality_area_transformer),
    ("regressor", linear_model)
])
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
pipeline_score_2 = cross_val_score(pipeline_2, Ames[["OverallQual", "GrLivArea"]], y,
                                   cv=5).mean()

# Output the mean CV scores rounded to four decimal places
# 打印输出 / Print output
print("Example With Pipeline, Mean CV R^2 score for 'Quality Weighted Area': "
      "{:.3f}".format(pipeline_score_2))
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Pipeline



---

### Imputation



---

### Chapter Summary / 章节总结

# Chapter 08 Summary / 第08章总结

## Theme / 主题: Chapter 08 / Chapter 08

This chapter contains **4 code files** demonstrating chapter 08.

本章包含 **4 个代码文件**，演示Chapter 08。

---
## Evolution / 演化路线

  1. `01_cv.ipynb` — Cv
  2. `02_synthetic.ipynb` — Synthetic
  3. `03_pipeline.ipynb` — Pipeline
  4. `04_imputation.ipynb` — Imputation

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 08) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 08）是机器学习流水线中的基础构建块。

---
