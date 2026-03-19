import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector

Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
X = Ames.drop("SalePrice", axis=1)
y = Ames["SalePrice"]
model = LinearRegression()

# Apply Sequential Feature Selector with tolerance = 0.005
sfs_tol = SequentialFeatureSelector(model, n_features_to_select="auto", tol=0.005)
sfs_tol.fit(X, y)

# Get the number of features selected with tolerance
n_features_selected = sum(sfs_tol.get_support())

# Prepare to store the mean CV R^2 scores for each number of features
mean_scores_tol = []

# Iterate over a range from 1 feature to the Sweet Spot
for n_features_to_select in range(1, n_features_selected + 1):
    sfs = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select)
    sfs.fit(X, y)
    selected_features = X.columns[sfs.get_support()]
    score = cross_val_score(model, X[selected_features], y, cv=5, scoring="r2").mean()
    mean_scores_tol.append(score)

# Print the selected features and their performance
selected_features = X.columns[sfs_tol.get_support()]
print(f"Number of features selected: {n_features_selected}")
print(f"Selected features: {selected_features.tolist()}")
print(f"Mean CV R^2 Score using SFS with tol=0.005: {mean_scores_tol[-1]:.4f}")
