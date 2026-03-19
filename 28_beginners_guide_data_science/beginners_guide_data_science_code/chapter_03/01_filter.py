import pandas as pd

# Load the dataset
Ames = pd.read_csv('Ames.csv')

# Simple querying: Select houses priced above $600,000
high_value_houses = Ames.query('SalePrice > 600000')
print(high_value_houses)
