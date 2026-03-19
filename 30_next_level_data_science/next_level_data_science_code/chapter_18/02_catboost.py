# Import libraries to run CatBoost Regressor
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score

# Load dataset
data = pd.read_csv("Ames.csv")
X = data.drop(["SalePrice"], axis=1)
y = data["SalePrice"]

# Identify and fill NaNs in categorical columns
cat_features = [col for col in X.columns if X[col].dtype == "object"]
X["Electrical"] = X["Electrical"].fillna(X["Electrical"].mode()[0])
X[cat_features] = X[cat_features].fillna("Missing")

# Identify categorical columns
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# Define and train the default CatBoost model
default_model = CatBoostRegressor(cat_features=cat_features, random_state=42, verbose=0)
default_scores = cross_val_score(default_model, X, y, cv=5, scoring="r2")
print(f"Average R^2 score for default CatBoost: {default_scores.mean():.4f}")

# Define and train the CatBoost model with ordered boosting
ordered_model = CatBoostRegressor(cat_features=cat_features, random_state=42,
                                  boosting_type="Ordered", verbose=0)
ordered_scores = cross_val_score(ordered_model, X, y, cv=5, scoring="r2")
print("Average R^2 score for CatBoost with ordered boosting: "
      f"{ordered_scores.mean():.4f}")
