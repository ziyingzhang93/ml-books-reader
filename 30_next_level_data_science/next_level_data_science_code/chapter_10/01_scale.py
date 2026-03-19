# Import necessary libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, Ridge, ElasticNet

# Load the dataset and remove columns with missing values
Ames = pd.read_csv("Ames.csv").dropna(axis=1)

# Identify numeric and categorical features, excluding "PID" and "SalePrice"
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns
X = Ames[numeric_features.tolist() + categorical_features.tolist()]

# Target variable
y = Ames["SalePrice"]

# Pipeline for numeric features
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

# Pipeline for categorical features
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combined preprocessor for both numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

# Define the model pipelines with preprocessor and regressor
pipelines = {
    "Lasso": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Lasso(max_iter=20000))]),
    "Ridge": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Ridge())]),
    "ElasticNet": Pipeline(steps=[("preprocessor", preprocessor),
                                  ("regressor", ElasticNet())])
}

# Perform cross-validation and store results in a dictionary
cv_results = {}
for name, pipeline in pipelines.items():
    scores = cross_val_score(pipeline, X, y)
    cv_results[name] = round(scores.mean(), 4)

# Output the mean cross-validation scores
print(cv_results)
