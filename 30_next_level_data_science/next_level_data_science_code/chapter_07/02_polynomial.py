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
