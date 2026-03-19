# Load the Ames dataset
import pandas as pd
Ames = pd.read_csv('Ames.csv')

# Reassign data type
Ames['MSSubClass'] = Ames['MSSubClass'].astype('object')
Ames['YrSold'] = Ames['YrSold'].astype('object')
Ames['MoSold'] = Ames['MoSold'].astype('object')

# Determine the data type for each feature after conversion
data_types = Ames.dtypes

# Tally the total by data type
type_counts = data_types.value_counts()

print(type_counts)
