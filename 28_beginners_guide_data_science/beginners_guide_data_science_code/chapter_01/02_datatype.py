import pandas as pd
Ames = pd.read_csv('Ames.csv')

# Determine the data type for each feature
data_types = Ames.dtypes

# Tally the total by data type
type_counts = data_types.value_counts()
print(type_counts)
