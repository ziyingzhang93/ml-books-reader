import pandas as pd

Ames = pd.read_csv('Ames.csv')
top_10_expensive_homes = Ames.nlargest(10, 'SalePrice')
features = ['SalePrice', 'GrLivArea', 'OverallQual', 'KitchenQual', 'TotRmsAbvGrd',
            'Fireplaces']
top_10_df = top_10_expensive_homes[features]
elite = top_10_df.query('OverallQual == 10 & KitchenQual == "Ex" & Fireplaces >= 2') \
                 .copy()

# Introduce PSF to rank the options
elite['PSF'] = elite['SalePrice']/elite['GrLivArea']
print(elite.sort_values(by='PSF'))
