import pandas as pd

Ames = pd.read_csv('Ames.csv')

# Cross check entire homes to search for better value
Ames['PSF'] = Ames['SalePrice']/Ames['GrLivArea']
value = Ames.query('PSF < 175 & OverallQual == 10 & KitchenQual == "Ex" & Fireplaces >=2')
print(value[['SalePrice', 'GrLivArea', 'OverallQual', 'KitchenQual', 'TotRmsAbvGrd',
             'Fireplaces', 'PSF']])
