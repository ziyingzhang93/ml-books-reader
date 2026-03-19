import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder

Ames = pd.read_csv("Ames.csv").select_dtypes(include=["object"]).dropna(axis=1)
y = pd.read_csv("Ames.csv")["SalePrice"]

feature_scores = {}
for feature in Ames.columns:
    encoder = OneHotEncoder(drop="first")
    X_encoded = encoder.fit_transform(Ames[[feature]])
    model = LinearRegression()
    scores = cross_val_score(model, X_encoded, y)
    mean_score = scores.mean()
    feature_scores[feature] = mean_score

sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)

print("Top 5 Categorical Features:")
for feature, score in sorted_features[0:5]:
    print(f"{feature}: Mean CV R^2 = {score:.4f}")
