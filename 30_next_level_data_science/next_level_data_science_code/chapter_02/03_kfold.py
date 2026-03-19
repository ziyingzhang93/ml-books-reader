import pandas as pd
# Import k-fold and necessary libraries
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

Ames = pd.read_csv("Ames.csv")

# Select features and target
X = Ames[['GrLivArea']].values  # Convert to numpy array for KFold
y = Ames['SalePrice'].values    # Convert to numpy array for KFold

# Initialize linear regression and k-fold
model = LinearRegression()
kf = KFold(n_splits=5)

# k-fold cross-validation in detailed steps
for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
    # Split the data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the model and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate and print the R^2 score for the current fold
    print(f"Fold {fold}:")
    print(f"TRAIN set size: {len(train_index)}")
    print(f"TEST set size: {len(test_index)}")
    print(f"R^2 score: {round(r2_score(y_test, y_pred), 4)}\n")
