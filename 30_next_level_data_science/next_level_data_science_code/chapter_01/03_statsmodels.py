import pandas as pd
import statsmodels.api as sm

# Load the Ames dataset
Ames = pd.read_csv("Ames.csv")

# Select features and target
X = Ames[["GrLivArea"]]  # Feature: GrLivArea, 2D matrix
y = Ames["SalePrice"]    # Target: SalePrice, 1D vector

# statsmodels requires to add a constant explicitly to model the intercept
X_with_constant = sm.add_constant(X)

# Fit the OLS model
model_stats = sm.OLS(y, X_with_constant).fit()

# Print the summary of the model
print(model_stats.summary())
