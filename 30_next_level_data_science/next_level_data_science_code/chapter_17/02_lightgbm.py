# Import libraries to run LightGBM
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

# Load the Ames Housing Dataset
data = pd.read_csv("Ames.csv")
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]

# Convert categorical columns to "category" dtype
categorical_cols = X.select_dtypes(include=["object"]).columns
X[categorical_cols] = X[categorical_cols].apply(lambda x: x.astype("category"))

# Define the default GBDT model
gbdt_model = lgb.LGBMRegressor(verbose=-1)
gbdt_scores = cross_val_score(gbdt_model, X, y, cv=5)
print(f"Average R^2 score for default Light GBM (with GBDT): {gbdt_scores.mean():.4f}")

# Define the GOSS model
goss_model = lgb.LGBMRegressor(boosting_type="goss", verbose=-1)
goss_scores = cross_val_score(goss_model, X, y, cv=5)
print(f"Average R^2 score for Light GBM with GOSS: {goss_scores.mean():.4f}")
