import pandas as pd

Ames = pd.read_csv('Ames.csv')

# Advanced querying: Select houses with more than 3 bedrooms and priced below $300,000
specific_houses = Ames.query('BedroomAbvGr > 3 & SalePrice < 300000')
print(specific_houses)
