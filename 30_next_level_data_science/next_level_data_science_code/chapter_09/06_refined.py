import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# Load the data
Ames = pd.read_csv('Ames.csv')
features = ['GrLivArea', '1stFlrSF', 'LowQualFinSF']  # Remove '2ndFlrSF' after Lasso
X = Ames[features]
y = Ames['SalePrice']

# Initialize a K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Collect coefficients and CV scores
coefficients = []
cv_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Initialize and fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    coefficients.append(model.coef_)

    # Calculate R^2 score using the model's score method
    score = model.score(X_test, y_test)
    # print(score)
    cv_scores.append(score)

# Plotting the coefficients
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.boxplot(np.array(coefficients), labels=features)
plt.title('Box Plot of Coefficients Across Folds (MLR)')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.grid(True)

# Plotting the CV scores
plt.subplot(1, 2, 2)
plt.plot(range(1, 6), cv_scores, marker='o', linestyle='-')  # make x-axis to start from 1
plt.title('Cross-Validation R^2 Scores (MLR)')
plt.xlabel('Fold')
plt.xticks(range(1, 6))  # Set x-ticks to match fold numbers
plt.ylabel('R^2 Score')
plt.ylim(min(cv_scores) - 0.05, max(cv_scores) + 0.05)  # Dynamically adjust y-axis limits
plt.grid(True)

# Annotate mean R^2 score
mean_r2 = np.mean(cv_scores)
plt.annotate(f'Mean CV R^2: {mean_r2:.3f}', xy=(1.25, 0.65), color='red', fontsize=14)

plt.tight_layout()
plt.show()
