# Import necessary libraries
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load data
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

# Prepare the BsmtQual imputation and encoding within a nested pipeline
bsmt_qual_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Setup ColumnTransformer for all preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cubic", cubic_transformer, ["OverallQual"]),
        ("quality_area_transform", quality_area_transformer,
            ["OverallQual", "GrLivArea"]),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"),
            ["Neighborhood", "ExterQual", "KitchenQual"]),
        ("bsmt_qual", bsmt_qual_transformer, ["BsmtQual"]),  # Adding BsmtQual handling
        ("passthrough", "passthrough", ["YearBuilt"])
    ])

# Create the pipeline with the preprocessor and linear regression
pipeline_4 = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", linear_model)
])

# Evaluate the pipeline using 5-fold cross-validation
pipeline_score = cross_val_score(pipeline_4, Ames, y, cv=5).mean()

# Output the mean CV scores rounded to four decimal places
print("Mean CV R^2 score with imputing & transformations: {:.3f}".format(pipeline_score))
