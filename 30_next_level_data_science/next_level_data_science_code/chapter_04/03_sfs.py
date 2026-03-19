import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector

Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
X = Ames.drop("SalePrice", axis=1)
y = Ames["SalePrice"]
model = LinearRegression()

# Perform Sequential Feature Selector with n=5
sfs = SequentialFeatureSelector(model, n_features_to_select=5)
sfs.fit(X, y)

selected_features = X.columns[sfs.get_support()].to_list()
print(f"Features selected by SFS: {selected_features}")

scores = cross_val_score(model, Ames[selected_features], y)
print(f"Mean CV R^2 Score using SFS with n=5: {scores.mean():.4f}")
