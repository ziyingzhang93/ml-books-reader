import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
import matplotlib.pyplot as plt

Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
X = Ames.drop("SalePrice", axis=1)
y = Ames["SalePrice"]
model = LinearRegression()

# Prepare to store the mean CV R^2 scores for each number of features
mean_scores = []

# Performance of SFS from 1 feature to the maximum number of features available
for n_features_to_select in range(1, len(X.columns)):
    sfs = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select)
    sfs.fit(X, y)
    selected_features = X.columns[sfs.get_support()]
    score = cross_val_score(model, X[selected_features], y, cv=5, scoring="r2").mean()
    mean_scores.append(score)

# Plot the mean CV R^2 scores against the number of features selected
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(X.columns)), mean_scores, marker="o")
plt.title("Performance vs. Number of Features Selected")
plt.xlabel("Number of Features")
plt.ylabel("Mean CV R^2 Score")
plt.grid(True)
plt.show()
