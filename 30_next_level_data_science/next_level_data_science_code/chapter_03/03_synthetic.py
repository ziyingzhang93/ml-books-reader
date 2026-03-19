import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
X = Ames.drop("SalePrice", axis=1)  # Features
y = Ames["SalePrice"]  # Target variable

# Create a new feature
Ames['QualityArea'] = Ames['OverallQual'] * Ames['GrLivArea']

# Setting up the feature and target variable for the new 'QualityArea' feature
X = Ames[['QualityArea']]  # New feature
y = Ames['SalePrice']

# 5-Fold CV on Linear Regression
model = LinearRegression()
cv_scores = cross_val_score(model, X, y, cv=5)

# Calculating the mean of the CV scores
mean_cv_score = cv_scores.mean()
print(f"Mean CV R^2 score using 'Quality Weighted Area': {mean_cv_score:.4f}")
