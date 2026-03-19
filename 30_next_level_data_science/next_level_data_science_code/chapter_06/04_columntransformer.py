# Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load data
Ames = pd.read_csv("Ames.csv")

# Select features and target
features = Ames[["GrLivArea", "Neighborhood"]]
target = Ames["SalePrice"]

# Preprocess features using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", ["GrLivArea"]),
        ("cat", OneHotEncoder(sparse_output=False, drop=["MeadowV"],
                              handle_unknown="ignore"), ["Neighborhood"])
    ])

# Fit and transform the features
X_transformed = preprocessor.fit_transform(features)
feature_names = ["GrLivArea"] + \
                list(preprocessor.named_transformers_["cat"].get_feature_names_out())

# Initialize KFold
kf = KFold(n_splits=5)

# Initialize variables to store results
coefficients_list = []
intercepts_list = []
scores = []

# Perform the KFold cross-validation
for train_index, test_index in kf.split(X_transformed):
    X_train, X_test = X_transformed[train_index], X_transformed[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]

    # Initialize the linear regression model
    model = LinearRegression()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Store coefficients and intercepts
    coefficients_list.append(model.coef_)
    intercepts_list.append(model.intercept_)

    # Evaluate the model
    scores.append(model.score(X_test, y_test))

# Calculate the mean of scores, coefficients, and intercepts
average_score = np.mean(scores)
average_coefficients = np.mean(coefficients_list, axis=0)
average_intercept = np.mean(intercepts_list)

# Display the average R^2 score and y-intercept across all folds
# The y-intercept is the baseline price in "MeadowV" with no additional living area
print(f"Mean CV R^2 Score of Combined Model: {average_score:.4f}")
print(f"Mean y-intercept = {average_intercept:.0f}")

# Create a DataFrame for the coefficients
df_coefficients = pd.DataFrame({
    "Feature": feature_names,
    "Average Coefficient": average_coefficients
    }).sort_values(by="Average Coefficient").reset_index(drop=True)

# Display the DataFrame
print("Coefficients for Combined Model:")
print(df_coefficients)
