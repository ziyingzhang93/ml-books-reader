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
