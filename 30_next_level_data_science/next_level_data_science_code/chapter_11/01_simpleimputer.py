# Import the necessary libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import cross_val_score

# Load the dataset
Ames = pd.read_csv("Ames.csv")

# Exclude "PID" and "SalePrice" from features and handle the "Electrical" column
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])
electrical_feature = ["Electrical"]  # Specifically handle the "Electrical" column

# Helper function to fill "Missing" for missing categorical data
def fill_none(X):
    return X.fillna("Missing")

# Pipeline for numeric features: Impute missing values then scale
numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Pipeline for general categorical features: Fill missing values with "Missing" then
# apply one-hot encoding
categorical_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Specific transformer for "Electrical" using the mode for imputation
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("onehot_electrical", OneHotEncoder(handle_unknown="ignore"))
])

# Combined preprocessor for numeric, general categorical, and electrical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("electrical", electrical_transformer, electrical_feature)
    ])

# Target variable
y = Ames["SalePrice"]

# All features
X = Ames[numeric_features.tolist() + categorical_features.tolist() + electrical_feature]

# Define the model pipelines with preprocessor and regressor
models = {
    "Lasso": Lasso(max_iter=20000),
    "Ridge": Ridge(),
    "ElasticNet": ElasticNet()
}

results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    # Perform cross-validation
    scores = cross_val_score(pipeline, X, y)
    results[name] = round(scores.mean(), 4)

# Output the cross-validation scores
print("Cross-validation scores with Simple Imputer:", results)
