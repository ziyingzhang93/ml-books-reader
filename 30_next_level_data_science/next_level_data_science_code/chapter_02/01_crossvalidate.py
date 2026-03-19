# Load the Ames dataset
import pandas as pd
Ames = pd.read_csv("Ames.csv")

# Import Linear Regression, Train-Test, Cross-Validation from scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score

# Select features and target
X = Ames[["GrLivArea"]]  # Feature: GrLivArea, a 2D matrix
y = Ames["SalePrice"]    # Target: SalePrice, a 1D vector

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model using Train-Test
model = LinearRegression()
model.fit(X_train, y_train)
train_test_score = round(model.score(X_test, y_test), 4)
print(f"Train-Test R^2 Score: {train_test_score}")

# Perform 5-Fold Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=5)
cv_scores_rounded = [round(score, 4) for score in cv_scores]
print(f"Cross-Validation R^2 Scores: {cv_scores_rounded}")
