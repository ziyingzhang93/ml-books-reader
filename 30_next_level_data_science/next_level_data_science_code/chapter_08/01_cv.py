# Import necessary libraries
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Prepare data and setup for linear regression
Ames = pd.read_csv("Ames.csv")
y = Ames["SalePrice"]
linear_model = LinearRegression()

# Perform 5-fold cross-validation without Pipeline
cv_score = cross_val_score(linear_model, Ames[["OverallQual"]], y).mean()
print("Example Without Pipeline, Mean CV R^2 score for 'OverallQual': {:.3f}"
      .format(cv_score))

# Perform 5-fold cross-validation WITH Pipeline
pipeline = Pipeline([("regressor", linear_model)])
pipeline_score = cross_val_score(pipeline, Ames[["OverallQual"]], y, cv=5).mean()
print("Example With Pipeline, Mean CV R^2 for 'OverallQual': {:.3f}"
      .format(pipeline_score))
