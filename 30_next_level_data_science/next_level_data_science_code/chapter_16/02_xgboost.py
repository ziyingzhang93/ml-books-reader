# Import XGBoost to demonstrate native handling of missing values
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score

# Load the dataset
Ames = pd.read_csv("Ames.csv")

# Select numeric features with missing values
cols_with_missing = Ames.isnull().any()
X = Ames.loc[:, cols_with_missing].select_dtypes(include=["int", "float"])
y = Ames["SalePrice"]

# Check and print the total number of missing values
total_missing_values = X.isna().sum().sum()
print(f"Total number of missing values: {total_missing_values}")

# Initialize XGBoost regressor with default settings, fixed seed for reproducibility
xgb_model = xgb.XGBRegressor(seed=42)

# Perform 5-fold cross-validation
scores = cross_val_score(xgb_model, X, y, cv=5, scoring="r2")

# Calculate and display the average R-squared score
mean_r2 = scores.mean()
print(f"XGB with native imputing, average R^2 score: {mean_r2:.4f}")
