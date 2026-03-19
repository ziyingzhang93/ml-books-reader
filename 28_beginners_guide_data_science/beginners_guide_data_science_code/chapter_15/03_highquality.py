import pandas as pd

Ames = pd.read_csv('Ames.csv')
top_10_expensive_homes = Ames.nlargest(10, 'SalePrice')
features = ['SalePrice', 'GrLivArea', 'OverallQual', 'KitchenQual', 'TotRmsAbvGrd',
            'Fireplaces']
top_10_df = top_10_expensive_homes[features]

# Refine the search with highest quality, excellent kitchen, and 2 fireplaces
elite = top_10_df.query('OverallQual == 10 & KitchenQual == "Ex" & Fireplaces >= 2')
print(elite)
