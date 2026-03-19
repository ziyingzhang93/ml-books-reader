import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder

Ames = pd.read_csv("Ames.csv").select_dtypes(include=["object"]).dropna(axis=1)

# Set "SalePrice" as the target variable
y = pd.read_csv("Ames.csv")["SalePrice"]

# Dictionary to store feature names and their corresponding mean CV R^2 scores
feature_scores = {}

for feature in Ames.columns:
    encoder = OneHotEncoder(drop="first")
    X_encoded = encoder.fit_transform(Ames[[feature]])

    # Initialize the linear regression model
    model = LinearRegression()

    # Perform 5-fold cross-validation and calculate R^2 scores
    scores = cross_val_score(model, X_encoded, y)
    mean_score = scores.mean()

    # Store the mean R^2 score
    feature_scores[feature] = mean_score

# Sort features based on their mean CV R^2 scores in descending order
sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
print("Feature selected for highest predictability:", sorted_features[0][0])
