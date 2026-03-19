import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
X = Ames.drop("SalePrice", axis=1)  # Features
y = Ames["SalePrice"]  # Target variable
model = LinearRegression()

# Dictionary to hold feature names and their corresponding mean CV R^2 scores
feature_scores = {}

# Iterate over each feature, perform CV, and store the mean R^2 score
for feature in X.columns:
    X_single = X[[feature]]
    cv_scores = cross_val_score(model, X_single, y, cv=5)
    feature_scores[feature] = cv_scores.mean()

# Sort features based on their mean CV R^2 scores in descending order
sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)

# Print the top 3 features and their scores
top_3 = sorted_features[0:3]
for feature, score in top_3:
    print(f"Feature: {feature}, Mean CV R^2: {score:.4f}")
