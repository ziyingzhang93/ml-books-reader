import pandas as pd
Ames = pd.read_csv('Ames.csv')

# Determine the data type for each feature
data_types = Ames.dtypes

# View a few datatypes from the dataset (first and last 5 features)
print(data_types)
