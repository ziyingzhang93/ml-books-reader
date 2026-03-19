import pandas as pd
# Import Linear Regression from scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the Ames dataset
Ames = pd.read_csv("Ames.csv")

# Select features and target
X = Ames[["GrLivArea"]]  # Feature: GrLivArea, 2D matrix
y = Ames["SalePrice"]    # Target: SalePrice, 1D vector

# Split data into training (80%) and testing sets (20%)
# Setting the random state to a fixed number to make the output reproducible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Linear Regression model
model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)

# Scoring the model
score = round(model.score(X_test, y_test), 4)
print(f"Model R^2 Score: {score}")
