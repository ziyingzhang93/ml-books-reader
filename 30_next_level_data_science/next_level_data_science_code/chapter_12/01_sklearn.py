# Import the necessary libraries
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Load all the numeric features without any missing values
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"])
Ames.dropna(axis=1, inplace=True)
X = Ames.drop("SalePrice", axis=1)
y = Ames["SalePrice"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the decision tree
tree_model = DecisionTreeRegressor(max_depth=3)
tree_model.fit(X_train, y_train)

# Visualize the decision tree with sklearn
plt.figure(figsize=(20, 10))
tree.plot_tree(tree_model, feature_names=X.columns, filled=True,
               impurity=False, rounded=True, precision=2, fontsize=12)
plt.show()
