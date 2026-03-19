# Load the essential libraries and Ames dataset
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
X = Ames.drop("SalePrice", axis=1)
y = Ames["SalePrice"]

# Initialize the Linear Regression model
model = LinearRegression()

# Prepare to collect feature scores
feature_scores = {}

# Evaluate each feature with cross-validation
for feature in X.columns:
    X_single = X[[feature]]
    cv_scores = cross_val_score(model, X_single, y)
    feature_scores[feature] = cv_scores.mean()

# Identify the top 5 features based on mean CV R^2 scores
sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
top_5 = sorted_features[0:5]

# Display the top 5 features and their individual performance
for feature, score in top_5:
    print(f"Feature: {feature}, Mean CV R^2: {score:.4f}")
