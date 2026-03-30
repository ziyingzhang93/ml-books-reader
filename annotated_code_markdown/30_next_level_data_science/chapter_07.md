# 进阶数据科学 / Next Level Data Science
## Chapter 07

---

### Linear

# 01 — Linear / 线性模型

**Chapter 07 — File 1 of 3 / 第07章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Import the necessary libraries**.

本脚本演示 **Import the necessary libraries**。

---
## Step 1 — Import the necessary libraries

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
```

---
## Step 2 — Prepare data for linear regression

```python
Ames = pd.read_csv("Ames.csv")
X = Ames[["OverallQual"]]  # Predictor
y = Ames["SalePrice"]      # Response
```

---
## Step 3 — Create and fit the linear regression model

```python
linear_model = LinearRegression()
linear_model.fit(X, y)
```

---
## Step 4 — Coefficients

```python
intercept = int(linear_model.intercept_)
slope = int(linear_model.coef_[0])
eqn = f"Fitted Line: y = {slope}x - {abs(intercept)}"
```

---
## Step 5 — Perform 5-fold cross-validation to evaluate model performance

```python
cv_score = cross_val_score(linear_model, X, y).mean()
```

---
## Step 6 — Visualize Best Fit and display CV results

```python
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", alpha=0.5, label="Data points")
plt.plot(X, linear_model.predict(X), color="red", label=eqn)
plt.title("Linear Regression of SalePrice vs OverallQual", fontsize=16)
plt.xlabel("Overall Quality", fontsize=12)
plt.ylabel("Sale Price", fontsize=12)
plt.legend(fontsize=14)
plt.grid(True)
plt.text(1, 540000, f"5-Fold CV R^2: {cv_score:.3f}", fontsize=14, color="green")
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Import the necessary libraries 是机器学习中的常用技术。  
  *Import the necessary libraries is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Linear / 线性模型
# Complete Code / 完整代码
# ===============================

# Import the necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Prepare data for linear regression
Ames = pd.read_csv("Ames.csv")
X = Ames[["OverallQual"]]  # Predictor
y = Ames["SalePrice"]      # Response

# Create and fit the linear regression model
linear_model = LinearRegression()
linear_model.fit(X, y)

# Coefficients
intercept = int(linear_model.intercept_)
slope = int(linear_model.coef_[0])
eqn = f"Fitted Line: y = {slope}x - {abs(intercept)}"

# Perform 5-fold cross-validation to evaluate model performance
cv_score = cross_val_score(linear_model, X, y).mean()

# Visualize Best Fit and display CV results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", alpha=0.5, label="Data points")
plt.plot(X, linear_model.predict(X), color="red", label=eqn)
plt.title("Linear Regression of SalePrice vs OverallQual", fontsize=16)
plt.xlabel("Overall Quality", fontsize=12)
plt.ylabel("Sale Price", fontsize=12)
plt.legend(fontsize=14)
plt.grid(True)
plt.text(1, 540000, f"5-Fold CV R^2: {cv_score:.3f}", fontsize=14, color="green")
plt.show()
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Polynomial

# 02 — Polynomial / 02 Polynomial

**Chapter 07 — File 2 of 3 / 第07章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Import the necessary libraries**.

本脚本演示 **Import the necessary libraries**。

---
## Step 1 — Import the necessary libraries

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
```

---
## Step 2 — Load the data

```python
Ames = pd.read_csv("Ames.csv")
X = Ames[["OverallQual"]]
y = Ames["SalePrice"]
```

---
## Step 3 — Transform the predictor variable to polynomial features up to the 3rd degree

```python
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)
```

---
## Step 4 — Create and fit the polynomial regression model

```python
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
```

---
## Step 5 — Extract model coefficients that form the polynomial equation

```python
intercept = int(poly_model.intercept_)
coefs = np.rint(poly_model.coef_).astype(int)
eqn = f"Fitted Line: y = " \
      f"{coefs[0]}x^1 {coefs[1]:+d}x^2 {coefs[2]:+d}x^3 {intercept:+d}"
```

---
## Step 6 — Perform 5-fold cross-validation

```python
cv_score = cross_val_score(poly_model, X_poly, y).mean()
```

---
## Step 7 — Generate data to plot curve

```python
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
```

---
## Step 8 — Plot

```python
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", alpha=0.5, label="Data points")
plt.plot(X_range, poly_model.predict(X_range_poly), color="red", label=eqn)
plt.title("Polynomial Regression (3rd Degree) of SalePrice vs OverallQual", fontsize=16)
plt.xlabel("Overall Quality", fontsize=12)
plt.ylabel("Sale Price", fontsize=12)
plt.legend(fontsize=14)
plt.grid(True)
plt.text(1, 540000, f"5-Fold CV R^2: {cv_score:.3f}", fontsize=14, color="green")
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Import the necessary libraries 是机器学习中的常用技术。  
  *Import the necessary libraries is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Polynomial / 02 Polynomial
# Complete Code / 完整代码
# ===============================

# Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Load the data
Ames = pd.read_csv("Ames.csv")
X = Ames[["OverallQual"]]
y = Ames["SalePrice"]

# Transform the predictor variable to polynomial features up to the 3rd degree
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)

# Create and fit the polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Extract model coefficients that form the polynomial equation
intercept = int(poly_model.intercept_)
coefs = np.rint(poly_model.coef_).astype(int)
eqn = f"Fitted Line: y = " \
      f"{coefs[0]}x^1 {coefs[1]:+d}x^2 {coefs[2]:+d}x^3 {intercept:+d}"

# Perform 5-fold cross-validation
cv_score = cross_val_score(poly_model, X_poly, y).mean()

# Generate data to plot curve
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_range_poly = poly.transform(X_range)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", alpha=0.5, label="Data points")
plt.plot(X_range, poly_model.predict(X_range_poly), color="red", label=eqn)
plt.title("Polynomial Regression (3rd Degree) of SalePrice vs OverallQual", fontsize=16)
plt.xlabel("Overall Quality", fontsize=12)
plt.ylabel("Sale Price", fontsize=12)
plt.legend(fontsize=14)
plt.grid(True)
plt.text(1, 540000, f"5-Fold CV R^2: {cv_score:.3f}", fontsize=14, color="green")
plt.show()
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Cubic

# 03 — Cubic / 03 Cubic

**Chapter 07 — File 3 of 3 / 第07章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Import the necessary libraries**.

本脚本演示 **Import the necessary libraries**。

---
## Step 1 — Import the necessary libraries

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import FunctionTransformer
import matplotlib.pyplot as plt
```

---
## Step 2 — Load data

```python
Ames = pd.read_csv("Ames.csv")
X = Ames[["OverallQual"]]
y = Ames["SalePrice"]
```

---
## Step 3 — Function to apply cubic transformation

```python
def cubic_transformation(x):
    return x ** 3
```

---
## Step 4 — Apply transformation

```python
cubic_transformer = FunctionTransformer(cubic_transformation)
X_cubic = cubic_transformer.fit_transform(X)
```

---
## Step 5 — Fit model

```python
cubic_model = LinearRegression()
cubic_model.fit(X_cubic, y)
```

---
## Step 6 — Get coefficients and intercept

```python
intercept_cubic = int(cubic_model.intercept_)
coef_cubic = int(cubic_model.coef_[0])
eqn = f"Fitted Line: y = {coef_cubic}x^3 + {intercept_cubic}"
```

---
## Step 7 — Cross-validation

```python
cv_score_cubic = cross_val_score(cubic_model, X_cubic, y).mean()
```

---
## Step 8 — Generate data to plot curve

```python
X_range = np.linspace(X.min(), X.max(), 300)
X_range_cubic = cubic_transformer.transform(X_range)
```

---
## Step 9 — Plot

```python
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", alpha=0.5, label="Data points")
plt.plot(X_range, cubic_model.predict(X_range_cubic), color="red", label=eqn)
plt.title("Cubic Regression of SalePrice vs OverallQual", fontsize=16)
plt.xlabel("Overall Quality", fontsize=12)
plt.ylabel("Sale Price", fontsize=12)
plt.legend(fontsize=14)
plt.grid(True)
plt.text(1, 540000, f"5-Fold CV R^2: {cv_score_cubic:.3f}", fontsize=14, color="green")
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Import the necessary libraries 是机器学习中的常用技术。  
  *Import the necessary libraries is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cubic / 03 Cubic
# Complete Code / 完整代码
# ===============================

# Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import FunctionTransformer
import matplotlib.pyplot as plt

# Load data
Ames = pd.read_csv("Ames.csv")
X = Ames[["OverallQual"]]
y = Ames["SalePrice"]

# Function to apply cubic transformation
def cubic_transformation(x):
    return x ** 3

# Apply transformation
cubic_transformer = FunctionTransformer(cubic_transformation)
X_cubic = cubic_transformer.fit_transform(X)

# Fit model
cubic_model = LinearRegression()
cubic_model.fit(X_cubic, y)

# Get coefficients and intercept
intercept_cubic = int(cubic_model.intercept_)
coef_cubic = int(cubic_model.coef_[0])
eqn = f"Fitted Line: y = {coef_cubic}x^3 + {intercept_cubic}"

# Cross-validation
cv_score_cubic = cross_val_score(cubic_model, X_cubic, y).mean()

# Generate data to plot curve
X_range = np.linspace(X.min(), X.max(), 300)
X_range_cubic = cubic_transformer.transform(X_range)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", alpha=0.5, label="Data points")
plt.plot(X_range, cubic_model.predict(X_range_cubic), color="red", label=eqn)
plt.title("Cubic Regression of SalePrice vs OverallQual", fontsize=16)
plt.xlabel("Overall Quality", fontsize=12)
plt.ylabel("Sale Price", fontsize=12)
plt.legend(fontsize=14)
plt.grid(True)
plt.text(1, 540000, f"5-Fold CV R^2: {cv_score_cubic:.3f}", fontsize=14, color="green")
plt.show()
```

---

### Chapter Summary / 章节总结

# Chapter 07 Summary / 第07章总结

## Theme / 主题: Chapter 07 / Chapter 07

This chapter contains **3 code files** demonstrating chapter 07.

本章包含 **3 个代码文件**，演示Chapter 07。

---
## Evolution / 演化路线

  1. `01_linear.ipynb` — Linear
  2. `02_polynomial.ipynb` — Polynomial
  3. `03_cubic.ipynb` — Cubic

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 07) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 07）是机器学习流水线中的基础构建块。

---
