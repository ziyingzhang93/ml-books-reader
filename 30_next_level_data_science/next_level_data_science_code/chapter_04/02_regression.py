import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
X = Ames.drop("SalePrice", axis=1)
y = Ames["SalePrice"]

# Evaluate each feature to find the top 5
model = LinearRegression()
feature_scores = {}
for feature in X.columns:
    X_single = X[[feature]]
    cv_scores = cross_val_score(model, X_single, y)
    feature_scores[feature] = cv_scores.mean()

sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
top_5 = sorted_features[0:5]

# Extracting the top 5 features for our multiple linear regression
top_features = [feature for feature, score in top_5]

# Building the model with the top 5 features
X_top = Ames[top_features]

# Evaluating the model with cross-validation
cv_scores_mlr = cross_val_score(model, X_top, y, cv=5, scoring="r2")
mean_mlr_score = cv_scores_mlr.mean()

print(f"Mean CV R^2 Score for Multiple Linear Regression Model: {mean_mlr_score:.4f}")
