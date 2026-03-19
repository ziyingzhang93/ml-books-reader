# Experiment with Leaf-wise Tree Growth
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

# Define a range of leaf sizes to test
leaf_sizes = [5, 10, 15, 31, 50, 100]

# Results storage
results = {}

# Experiment with different leaf sizes for GBDT
results["GBDT"] = {}
print('Testing different "num_leaves" for GBDT:')
for leaf_size in leaf_sizes:
    model = lgb.LGBMRegressor(boosting_type="gbdt", num_leaves=leaf_size, verbose=-1)
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    results["GBDT"][leaf_size] = scores.mean()
    print(f"num_leaves = {leaf_size}: Average R^2 score = {scores.mean():.4f}")

# Experiment with different leaf sizes for GOSS
results["GOSS"] = {}
print('\nTesting different "num_leaves" for GOSS:')
for leaf_size in leaf_sizes:
    model = lgb.LGBMRegressor(boosting_type="goss", num_leaves=leaf_size, verbose=-1)
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    results["GOSS"][leaf_size] = scores.mean()
    print(f"num_leaves = {leaf_size}: Average R^2 score = {scores.mean():.4f}")
