# Perform Cross-Validated Recursive Feature Elimination for XGB
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score

# Load the dataset
Ames = pd.read_csv("Ames.csv")

# Convert selected features to "object" type to treat them as categorical
for col in ["MSSubClass", "YrSold", "MoSold"]:
    Ames[col] = Ames[col].astype("object")

# Convert all object-type features to categorical and then to codes
categorical_features = Ames.select_dtypes(include=["object"]).columns
for col in categorical_features:
    Ames[col] = Ames[col].astype("category").cat.codes

# Optional: Show that the categorical data encoded into integers
# print(Ames)

# Select features and target
X = Ames.drop(columns=["SalePrice", "PID"])
y = Ames["SalePrice"]

# Initialize XGBoost regressor
xgb_model = xgb.XGBRegressor(seed=42, enable_categorical=True)

# Initialize RFECV
rfecv = RFECV(estimator=xgb_model, step=1, cv=5,
              scoring="r2", min_features_to_select=1)

# Fit RFECV
rfecv.fit(X, y)

# Print the optimal number of features and their names
print("Optimal number of features: ", rfecv.n_features_)
print("Best features: ", X.columns[rfecv.support_])
