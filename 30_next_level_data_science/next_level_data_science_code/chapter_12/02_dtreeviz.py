# Import the necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import dtreeviz

# Load all the numeric features without any missing values
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"])
Ames.dropna(axis=1, inplace=True)
X = Ames.drop("SalePrice", axis=1)
y = Ames["SalePrice"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Decision Tree
tree_model = DecisionTreeRegressor(max_depth=3)
tree_model.fit(X_train.values, y_train)

# Visualize the decision tree using dtreeviz
viz = dtreeviz.model(tree_model, X_train, y_train,
               target_name="SalePrice", feature_names=X_train.columns.tolist())

# In Jupyter Notebook, you can directly view the visual using the below:
# viz.view()  # Renders and displays the SVG visualization

# In PyCharm, you can render and display the SVG image:
v = viz.view()     # render as SVG into internal object
v.show()           # pop up window
