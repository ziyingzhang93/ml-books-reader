import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
# Import OneHotEncoder to preprocess a categorical feature
from sklearn.preprocessing import OneHotEncoder

Ames = pd.read_csv("Ames.csv")

# One-hot encoding for "Neighborhood", Note: drop=["MeadowV"]
encoder = OneHotEncoder(sparse=False, drop=["MeadowV"])
X = encoder.fit_transform(Ames[["Neighborhood"]])
y = Ames["SalePrice"].values

# Setup KFold and initialize storage
kf = KFold(n_splits=5)
scores = []
coefficients = []
intercept = []

# Perform the KFold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Append the results for each fold
    scores.append(model.score(X_test, y_test))
    coefficients.append(model.coef_)
    intercept.append(model.intercept_)

mean_score = np.mean(scores)
print(f"Mean CV R^2 = {mean_score:.4f}")
mean_coefficients = np.mean(coefficients, axis=0)
mean_intercept = np.mean(intercept)
print(f"Mean y-intercept = {mean_intercept:.0f}")

# Retrieve neighborhood names from the encoder, adjusting for the dropped category
neighborhoods = encoder.categories_[0]
if "MeadowV" in neighborhoods:
    neighborhoods = [name for name in neighborhoods if name != "MeadowV"]

# Create a DataFrame to nicely display neighborhoods with their average coefficients
coefficients_df = pd.DataFrame({
    "Neighborhood": neighborhoods,
    "Average Coefficient": mean_coefficients.round(0).astype(int)
})

# Print or return the DataFrame
print(coefficients_df.sort_values(by="Average Coefficient").reset_index(drop=True))
