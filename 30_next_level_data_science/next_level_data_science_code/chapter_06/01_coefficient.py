# Set up to obtain CV model performance and coefficient using k-fold
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

Ames = pd.read_csv("Ames.csv")
X = Ames[["GrLivArea"]].values  # get 2D matrix
y = Ames["SalePrice"].values    # get 1D vector

model = LinearRegression()
kf = KFold(n_splits=5)
coefs = []
scores = []

# Manually perform k-fold cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
    # Split the data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the model, obtain fold performance and coefficient
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
    coefs.append(model.coef_)

mean_score = np.mean(scores)
print(f"Mean CV R^2 = {mean_score:.4f}")

mean_coefs = np.mean(coefs)
print(f"Mean Coefficient = {mean_coefs:.4f}")
