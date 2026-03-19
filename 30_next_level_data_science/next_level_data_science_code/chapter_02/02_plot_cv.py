import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
# Import Seaborn and Matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

# Perform 5-fold cross-validation. Let cv_scores_rounded contains your
# cross-validation scores, and train_test_score is your single train-test R^2 score
Ames = pd.read_csv("Ames.csv")
X = Ames[["GrLivArea"]]  # Feature: GrLivArea, a 2D matrix
y = Ames["SalePrice"]    # Target: SalePrice, a 1D vector
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
train_test_score = round(model.score(X_test, y_test), 4)
cv_scores = cross_val_score(model, X, y, cv=5)
cv_scores_rounded = [round(score, 4) for score in cv_scores]

# Plot the box plot for cross-validation scores
cv_scores_df = pd.DataFrame(cv_scores_rounded, columns=["Cross-Validation Scores"])
sns.boxplot(data=cv_scores_df, y="Cross-Validation Scores",
            width=0.3, color="lightblue", fliersize=0)

# Overlay individual scores as points
plt.scatter([0] * len(cv_scores_rounded), cv_scores_rounded,
            color="blue", label="Cross-Validation Scores")
plt.scatter(0, train_test_score, color="red", zorder=5, label="Train-Test Score")

# Plot the visual
plt.title("Model Evaluation: Cross-Validation vs. Train-Test")
plt.ylabel("R^2 Score")
plt.xticks([0], ["Evaluation Scores"])
plt.legend(loc="lower left", bbox_to_anchor=(0, +0.1))
plt.show()
