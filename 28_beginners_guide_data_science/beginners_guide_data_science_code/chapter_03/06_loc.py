import pandas as pd

Ames = pd.read_csv('Ames.csv')

# This is a list of neighborhoods with higher average sale prices
high_value_neighborhoods = ['NridgHt', 'NoRidge', 'StoneBr']

# Use df.loc[] to select houses in high-value neighborhoods based on your conditions
high_value_houses = Ames.loc[(Ames['BedroomAbvGr'] > 3) &
                             (Ames['SalePrice'] < 300000) &
                             (Ames['Neighborhood'].isin(high_value_neighborhoods)),
                             ['Neighborhood', 'SalePrice', 'GrLivArea']]

print(high_value_houses.head())
