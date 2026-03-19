# Load the Ames dataset
import pandas as pd
Ames = pd.read_csv('Ames.csv')

# Using select_dtypes()
numerical_features = Ames.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = Ames.select_dtypes(include=['object', 'category']).columns.tolist()
print("Numerical features (int64 and float64):", numerical_features)
print("Categorical features (object and category):", categorical_features)

# Using describe() to automatically extract numerical features
numerical_features = Ames.describe().columns.tolist()
print("Numerical features from describe():", numerical_features)

# Data dictionary and domain knowledge could be useful in setting the threshold
threshold = 10
categorical_features = Ames.columns[Ames.nunique() <= threshold].tolist()
print("Categorical features based on unique values:", categorical_features)

# Using value_counts() on each column or feature
print("Value counts:")
for column in Ames.columns:
    print(Ames[column].value_counts())

# Using info() on the Ames Dataset
print("info():")
Ames.info()
