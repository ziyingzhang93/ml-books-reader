# Implement GridSearchCV on ElasticNet to obtain optimal parameters

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

# Define range of alpha for ElasticNet
alpha = list(range(1, 21, 1))  # Ranges from 1 to 20 in increments of 1

# Define range of L1 ratio for ElasticNet
l1_ratio = [0.05, 0.5, 0.95]

# Setup Grid Search for ElasticNet
elasticnet_grid = GridSearchCV(estimator=pipelines['ElasticNet'],
                               param_grid={'regressor__alpha': alpha,
                                           'regressor__l1_ratio': l1_ratio},
                               verbose=1)  # Prints out progress
elasticnet_grid.fit(X, y)

# Extract the best parameters and best score for ElasticNet
elasticnet_best_params = elasticnet_grid.best_params_
elasticnet_best_score = elasticnet_grid.best_score_

print(f"Best parameters for ElasticNet: {elasticnet_best_params}")
print(f"Best cross-validation score: {round(elasticnet_best_score, 4)}")
