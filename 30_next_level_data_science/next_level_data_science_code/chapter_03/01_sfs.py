# Load only the numeric columns from the Ames dataset
import pandas as pd
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"])

# Drop any columns with missing values
Ames = Ames.dropna(axis=1)

# Import Linear Regression and Sequential Feature Selector from scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector

# Initializing the Linear Regression model
model = LinearRegression()

# Perform Sequential Feature Selector
sfs = SequentialFeatureSelector(model, n_features_to_select=1)
X = Ames.drop("SalePrice", axis=1)  # Features
y = Ames["SalePrice"]  # Target variable
sfs.fit(X,y)           # Uses a default of cv=5
selected_feature = X.columns[sfs.get_support()]
print("Feature selected for highest predictability:", selected_feature[0])
