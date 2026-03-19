# Import the necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Prepare data for linear regression
Ames = pd.read_csv("Ames.csv")
X = Ames[["OverallQual"]]  # Predictor
y = Ames["SalePrice"]      # Response

# Create and fit the linear regression model
linear_model = LinearRegression()
linear_model.fit(X, y)

# Coefficients
intercept = int(linear_model.intercept_)
slope = int(linear_model.coef_[0])
eqn = f"Fitted Line: y = {slope}x - {abs(intercept)}"

# Perform 5-fold cross-validation to evaluate model performance
cv_score = cross_val_score(linear_model, X, y).mean()

# Visualize Best Fit and display CV results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", alpha=0.5, label="Data points")
plt.plot(X, linear_model.predict(X), color="red", label=eqn)
plt.title("Linear Regression of SalePrice vs OverallQual", fontsize=16)
plt.xlabel("Overall Quality", fontsize=12)
plt.ylabel("Sale Price", fontsize=12)
plt.legend(fontsize=14)
plt.grid(True)
plt.text(1, 540000, f"5-Fold CV R^2: {cv_score:.3f}", fontsize=14, color="green")
plt.show()
