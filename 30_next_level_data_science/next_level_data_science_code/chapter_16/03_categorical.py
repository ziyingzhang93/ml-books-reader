# Demonstrate native handling of categorical features
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score

# Load the dataset
Ames = pd.read_csv("Ames.csv")

# Convert specified categorical features to "category" type
for col in ["Neighborhood", "BldgType", "HouseStyle"]:
    Ames[col] = Ames[col].astype("category")

# Include some numeric features for a balanced model
selected_features = ["OverallQual", "GrLivArea", "YearBuilt", "TotalBsmtSF", "1stFlrSF",
                     "Neighborhood", "BldgType", "HouseStyle"]
X = Ames[selected_features]
y = Ames["SalePrice"]

# Initialize XGBoost regressor with native handling for categorical data
xgb_model = xgb.XGBRegressor(
    seed=42,
    enable_categorical=True
)

# Perform 5-fold cross-validation
scores = cross_val_score(xgb_model, X, y, cv=5, scoring="r2")

# Calculate the average R-squared score
mean_r2 = scores.mean()

print(f"Average model R^2 score with selected categorical features: {mean_r2:.4f}")
