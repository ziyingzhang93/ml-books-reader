# Implement GridSearchCV on Lasso to obtain optimal alpha

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

# Define range of alpha values for Lasso
alpha = list(range(1, 21, 1))  # Ranges from 1 to 20 in increments of 1

# Setup Grid Search for Lasso
lasso_grid = GridSearchCV(estimator=pipelines["Lasso"],
                          param_grid={"regressor__alpha": alpha},
                          verbose=1)  # Prints out progress
lasso_grid.fit(X, y)

# Extract the best alpha and best score Lasso
lasso_best_alpha = lasso_grid.best_params_["regressor__alpha"]
lasso_best_score = lasso_grid.best_score_

print(f"Best alpha for Lasso: {lasso_best_alpha}")
print(f"Best cross-validation score: {round(lasso_best_score, 4)}")
