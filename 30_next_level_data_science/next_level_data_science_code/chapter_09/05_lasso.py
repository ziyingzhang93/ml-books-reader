# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Load the data
Ames = pd.read_csv("Ames.csv")
features = ["GrLivArea", "1stFlrSF", "2ndFlrSF", "LowQualFinSF"]
X = Ames[features]
y = Ames["SalePrice"]

# Initialize a k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Prepare to collect results
results = {}

for alpha in [1, 2]:  # Loop through both alpha values
    coefficients = []
    cv_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize and fit the Lasso regression model
        lasso_model = Lasso(alpha=alpha, max_iter=20000)
        lasso_model.fit(X_train_scaled, y_train)
        coefficients.append(lasso_model.coef_)

        # Calculate R^2 score using the model's score method
        score = lasso_model.score(X_test_scaled, y_test)
        cv_scores.append(score)

    results[alpha] = (coefficients, cv_scores)

# Plotting the results
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
alphas = [1, 2]

for i, alpha in enumerate(alphas):
    coefficients, cv_scores = results[alpha]

    # Plotting the coefficients
    axes[i, 0].boxplot(np.array(coefficients), labels=features)
    axes[i, 0].set_title(f"Box Plot of Coefficients (Lasso with alpha={alpha})")
    axes[i, 0].set_xlabel("Features")
    axes[i, 0].set_ylabel("Coefficient Value")
    axes[i, 0].grid(True)

    # Plotting the CV scores
    axes[i, 1].plot(range(1, 6), cv_scores, marker="o", linestyle="-")
    axes[i, 1].set_title(f"Cross-Validation R^2 Scores (Lasso with alpha={alpha})")
    axes[i, 1].set_xlabel("Fold")
    axes[i, 1].set_xticks(range(1, 6))
    axes[i, 1].set_ylabel("R^2 Score")
    axes[i, 1].set_ylim(min(cv_scores) - 0.05, max(cv_scores) + 0.05)
    axes[i, 1].grid(True)
    mean_r2 = np.mean(cv_scores)
    axes[i, 1].annotate(f"Mean CV R^2: {mean_r2:.3f}", xy=(1.25, 0.65),
                        color="red", fontsize=12)

plt.tight_layout()
plt.show()
