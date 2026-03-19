# Implement GridSearchCV on Ridge to obtain optimal alpha

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV

# Load dataset and identify numeric and categorical features
Ames = pd.read_csv("Ames.csv").dropna(axis=1)
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns

# Features and target variables
X = Ames[numeric_features.tolist() + categorical_features.tolist()]
y = Ames["SalePrice"]

# Set up transformers and pipelines
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
pipelines = {
    "Lasso": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Lasso(max_iter=20000))]),
    "Ridge": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Ridge())]),
    "ElasticNet": Pipeline(steps=[("preprocessor", preprocessor),
                                  ("regressor", ElasticNet())])
}

# Define range of alpha for Ridge
alpha = list(range(1, 21, 1))  # Ranges from 1 to 20 in increments of 1

# Setup Grid Search for Ridge
ridge_grid = GridSearchCV(estimator=pipelines["Ridge"],
                          param_grid={"regressor__alpha": alpha},
                          verbose=1)  # Prints out progress
ridge_grid.fit(X, y)

# Extract the best alpha and best score for Ridge
ridge_best_alpha = ridge_grid.best_params_["regressor__alpha"]
ridge_best_score = ridge_grid.best_score_

print(f"Best alpha for Ridge: {ridge_best_alpha}")
print(f"Best cross-validation score: {round(ridge_best_score, 4)}")
