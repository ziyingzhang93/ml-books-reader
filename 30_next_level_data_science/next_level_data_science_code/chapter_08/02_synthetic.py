# Import necessary libraries
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Prepare data and setup for linear regression
Ames = pd.read_csv("Ames.csv")
y = Ames["SalePrice"]
linear_model = LinearRegression()

# Perform 5-fold cross-validation without Pipeline
Ames["OWA"] = Ames["OverallQual"] * Ames["GrLivArea"]
cv_score_2 = cross_val_score(linear_model, Ames[["OWA"]], y).mean()
print("Example Without Pipeline, Mean CV R^2 score for 'Quality Weighted Area':"
      "{:.3f}".format(cv_score_2))

# WITH Pipeline
# Define the transformation function for "QualityArea"
def create_quality_area(X):
    X["QualityArea"] = X["OverallQual"] * X["GrLivArea"]
    return X[["QualityArea"]].values

# Setup the FunctionTransformer using the function
quality_area_transformer = FunctionTransformer(create_quality_area)

# Pipeline using the engineered feature "QualityArea"
pipeline_2 = Pipeline([
    ("quality_area_transform", quality_area_transformer),
    ("regressor", linear_model)
])
pipeline_score_2 = cross_val_score(pipeline_2, Ames[["OverallQual", "GrLivArea"]], y,
                                   cv=5).mean()

# Output the mean CV scores rounded to four decimal places
print("Example With Pipeline, Mean CV R^2 score for 'Quality Weighted Area': "
      "{:.3f}".format(pipeline_score_2))
