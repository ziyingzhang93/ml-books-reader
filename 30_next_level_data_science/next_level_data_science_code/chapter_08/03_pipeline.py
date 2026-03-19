# Import necessary libraries
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

# Prepare data and setup for linear regression
Ames = pd.read_csv("Ames.csv")
y = Ames["SalePrice"]
linear_model = LinearRegression()

# Function to apply cubic transformation
def cubic_transformation(x):
    return x ** 3

# Function to create "QualityArea"
def create_quality_area(X):
    X["QualityArea"] = X["OverallQual"] * X["GrLivArea"]
    return X[["QualityArea"]].values

# Setup the FunctionTransformer for cubic and quality area transformations
cubic_transformer = FunctionTransformer(cubic_transformation)
quality_area_transformer = FunctionTransformer(create_quality_area)

# Setup ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cubic", cubic_transformer, ["OverallQual"]),
        ("quality_area_transform", quality_area_transformer,
            ["OverallQual", "GrLivArea"]),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"),
            ["Neighborhood", "ExterQual", "KitchenQual"]),
        ("passthrough", "passthrough", ["YearBuilt"])
    ])

# Create the pipeline with the preprocessor and linear regression
pipeline_3 = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", linear_model)
])

# Evaluate the pipeline using 5-fold cross-validation
pipeline_score_3 = cross_val_score(pipeline_3, Ames, y, cv=5).mean()

# Output the mean CV scores rounded to four decimal places
print("Mean CV R^2 score with enhanced transformations: {:.3f}".format(pipeline_score_3))
