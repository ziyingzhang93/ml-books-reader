import pandas as pd

Ames = pd.read_csv('Ames.csv')

# Identify the 10 most expensive homes based on SalePrice with key features
top_10_expensive_homes = Ames.nlargest(10, 'SalePrice')
features = ['SalePrice', 'GrLivArea', 'OverallQual', 'KitchenQual', 'TotRmsAbvGrd',
            'Fireplaces']
top_10_df = top_10_expensive_homes[features]
print(top_10_df)
