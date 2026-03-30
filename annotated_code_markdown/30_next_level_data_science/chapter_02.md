# 进阶数据科学 / Next Level Data Science
## Chapter 02

---

### Crossvalidate

# 01 — Crossvalidate / 01 Crossvalidate

**Chapter 02 — File 1 of 3 / 第02章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Load the Ames dataset**.

本脚本演示 **Load the Ames dataset**。

---
## Step 1 — Load the Ames dataset

```python
import pandas as pd
Ames = pd.read_csv("Ames.csv")
```

---
## Step 2 — Import Linear Regression, Train-Test, Cross-Validation from scikit-learn

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
```

---
## Step 3 — Select features and target

```python
X = Ames[["GrLivArea"]]  # Feature: GrLivArea, a 2D matrix
y = Ames["SalePrice"]    # Target: SalePrice, a 1D vector
```

---
## Step 4 — Split data into training and testing sets

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---
## Step 5 — Linear Regression model using Train-Test

```python
model = LinearRegression()
model.fit(X_train, y_train)
train_test_score = round(model.score(X_test, y_test), 4)
print(f"Train-Test R^2 Score: {train_test_score}")
```

---
## Step 6 — Perform 5-Fold Cross-Validation

```python
cv_scores = cross_val_score(model, X, y, cv=5)
cv_scores_rounded = [round(score, 4) for score in cv_scores]
print(f"Cross-Validation R^2 Scores: {cv_scores_rounded}")
```

---
## Learning Notes / 学习笔记

- **概念**: Load the Ames dataset 是机器学习中的常用技术。  
  *Load the Ames dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Crossvalidate / 01 Crossvalidate
# Complete Code / 完整代码
# ===============================

# Load the Ames dataset
import pandas as pd
Ames = pd.read_csv("Ames.csv")

# Import Linear Regression, Train-Test, Cross-Validation from scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score

# Select features and target
X = Ames[["GrLivArea"]]  # Feature: GrLivArea, a 2D matrix
y = Ames["SalePrice"]    # Target: SalePrice, a 1D vector

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model using Train-Test
model = LinearRegression()
model.fit(X_train, y_train)
train_test_score = round(model.score(X_test, y_test), 4)
print(f"Train-Test R^2 Score: {train_test_score}")

# Perform 5-Fold Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=5)
cv_scores_rounded = [round(score, 4) for score in cv_scores]
print(f"Cross-Validation R^2 Scores: {cv_scores_rounded}")
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Plot Cv

# 02 — Plot Cv / 02 Plot Cv

**Chapter 02 — File 2 of 3 / 第02章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Import Seaborn and Matplotlib**.

本脚本演示 **Import Seaborn and Matplotlib**。

---
## Step 1 — Step 1

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
```

---
## Step 2 — Import Seaborn and Matplotlib

```python
import seaborn as sns
import matplotlib.pyplot as plt
```

---
## Step 3 — Perform 5-fold cross-validation. Let cv_scores_rounded contains your
cross-validation scores, and train_test_score is your single train-test R^2 score

```python
Ames = pd.read_csv("Ames.csv")
X = Ames[["GrLivArea"]]  # Feature: GrLivArea, a 2D matrix
y = Ames["SalePrice"]    # Target: SalePrice, a 1D vector
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
train_test_score = round(model.score(X_test, y_test), 4)
cv_scores = cross_val_score(model, X, y, cv=5)
cv_scores_rounded = [round(score, 4) for score in cv_scores]
```

---
## Step 4 — Plot the box plot for cross-validation scores

```python
cv_scores_df = pd.DataFrame(cv_scores_rounded, columns=["Cross-Validation Scores"])
sns.boxplot(data=cv_scores_df, y="Cross-Validation Scores",
            width=0.3, color="lightblue", fliersize=0)
```

---
## Step 5 — Overlay individual scores as points

```python
plt.scatter([0] * len(cv_scores_rounded), cv_scores_rounded,
            color="blue", label="Cross-Validation Scores")
plt.scatter(0, train_test_score, color="red", zorder=5, label="Train-Test Score")
```

---
## Step 6 — Plot the visual

```python
plt.title("Model Evaluation: Cross-Validation vs. Train-Test")
plt.ylabel("R^2 Score")
plt.xticks([0], ["Evaluation Scores"])
plt.legend(loc="lower left", bbox_to_anchor=(0, +0.1))
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Import Seaborn and Matplotlib 是机器学习中的常用技术。  
  *Import Seaborn and Matplotlib is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot Cv / 02 Plot Cv
# Complete Code / 完整代码
# ===============================

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
# Import Seaborn and Matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

# Perform 5-fold cross-validation. Let cv_scores_rounded contains your
# cross-validation scores, and train_test_score is your single train-test R^2 score
Ames = pd.read_csv("Ames.csv")
X = Ames[["GrLivArea"]]  # Feature: GrLivArea, a 2D matrix
y = Ames["SalePrice"]    # Target: SalePrice, a 1D vector
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
train_test_score = round(model.score(X_test, y_test), 4)
cv_scores = cross_val_score(model, X, y, cv=5)
cv_scores_rounded = [round(score, 4) for score in cv_scores]

# Plot the box plot for cross-validation scores
cv_scores_df = pd.DataFrame(cv_scores_rounded, columns=["Cross-Validation Scores"])
sns.boxplot(data=cv_scores_df, y="Cross-Validation Scores",
            width=0.3, color="lightblue", fliersize=0)

# Overlay individual scores as points
plt.scatter([0] * len(cv_scores_rounded), cv_scores_rounded,
            color="blue", label="Cross-Validation Scores")
plt.scatter(0, train_test_score, color="red", zorder=5, label="Train-Test Score")

# Plot the visual
plt.title("Model Evaluation: Cross-Validation vs. Train-Test")
plt.ylabel("R^2 Score")
plt.xticks([0], ["Evaluation Scores"])
plt.legend(loc="lower left", bbox_to_anchor=(0, +0.1))
plt.show()
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Kfold

# 03 — Kfold / 03 Kfold

**Chapter 02 — File 3 of 3 / 第02章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Import k-fold and necessary libraries**.

本脚本演示 **Import k-fold and necessary libraries**。

---
## Step 1 — Step 1

```python
import pandas as pd
```

---
## Step 2 — Import k-fold and necessary libraries

```python
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

Ames = pd.read_csv("Ames.csv")
```

---
## Step 3 — Select features and target

```python
X = Ames[['GrLivArea']].values  # Convert to numpy array for KFold
y = Ames['SalePrice'].values    # Convert to numpy array for KFold
```

---
## Step 4 — Initialize linear regression and k-fold

```python
model = LinearRegression()
kf = KFold(n_splits=5)
```

---
## Step 5 — k-fold cross-validation in detailed steps

```python
for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
```

---
## Step 6 — Split the data into training and testing sets

```python
X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

---
## Step 7 — Fit the model and predict

```python
model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
```

---
## Step 8 — Calculate and print the R^2 score for the current fold

```python
print(f"Fold {fold}:")
    print(f"TRAIN set size: {len(train_index)}")
    print(f"TEST set size: {len(test_index)}")
    print(f"R^2 score: {round(r2_score(y_test, y_pred), 4)}\n")
```

---
## Learning Notes / 学习笔记

- **概念**: Import k-fold and necessary libraries 是机器学习中的常用技术。  
  *Import k-fold and necessary libraries is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Kfold / 03 Kfold
# Complete Code / 完整代码
# ===============================

import pandas as pd
# Import k-fold and necessary libraries
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

Ames = pd.read_csv("Ames.csv")

# Select features and target
X = Ames[['GrLivArea']].values  # Convert to numpy array for KFold
y = Ames['SalePrice'].values    # Convert to numpy array for KFold

# Initialize linear regression and k-fold
model = LinearRegression()
kf = KFold(n_splits=5)

# k-fold cross-validation in detailed steps
for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
    # Split the data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the model and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate and print the R^2 score for the current fold
    print(f"Fold {fold}:")
    print(f"TRAIN set size: {len(train_index)}")
    print(f"TEST set size: {len(test_index)}")
    print(f"R^2 score: {round(r2_score(y_test, y_pred), 4)}\n")
```

---

### Chapter Summary / 章节总结

# Chapter 02 Summary / 第02章总结

## Theme / 主题: Chapter 02 / Chapter 02

This chapter contains **3 code files** demonstrating chapter 02.

本章包含 **3 个代码文件**，演示Chapter 02。

---
## Evolution / 演化路线

  1. `01_crossvalidate.ipynb` — Crossvalidate
  2. `02_plot_cv.ipynb` — Plot Cv
  3. `03_kfold.ipynb` — Kfold

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 02) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 02）是机器学习流水线中的基础构建块。

---
