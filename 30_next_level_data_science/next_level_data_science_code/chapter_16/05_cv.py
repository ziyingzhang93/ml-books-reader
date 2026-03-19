import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score

# Load the dataset and convert features
Ames = pd.read_csv("Ames.csv")
for col in ["MSSubClass", "YrSold", "MoSold"]:
    Ames[col] = Ames[col].astype("object")
categorical_features = Ames.select_dtypes(include=["object"]).columns
for col in categorical_features:
    Ames[col] = Ames[col].astype("category").cat.codes

# RFECV on an XGBoost regressor
X = Ames.drop(columns=["SalePrice", "PID"])
y = Ames["SalePrice"]
xgb_model = xgb.XGBRegressor(seed=42, enable_categorical=True)
rfecv = RFECV(estimator=xgb_model, step=1, cv=5,
              scoring="r2", min_features_to_select=1)
rfecv.fit(X, y)

# Cross-validate the final model using only the selected features
final_model = xgb.XGBRegressor(seed=42, enable_categorical=True)
cv_scores = cross_val_score(final_model, X.iloc[:, rfecv.support_], y, cv=5, scoring="r2")

# Calculate the average R-squared score
mean_r2 = cv_scores.mean()

print(f"Average Cross-validated R^2 score with remaining features: {mean_r2:.4f}")
