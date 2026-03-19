import pandas as pd

Ames = pd.read_csv('Ames.csv')

high_value_neighborhoods = ['NridgHt', 'NoRidge', 'StoneBr']

# Filter for houses not in the 'high_value_neighborhoods',
# with at least 3 bedrooms above grade, and priced below $300,000
low_value_spacious = Ames.loc[(~Ames['Neighborhood'].isin(high_value_neighborhoods)) &
                              (Ames['BedroomAbvGr'] >= 3) &
                              (Ames['SalePrice'] < 300000)]

# Sort these houses by 'SalePrice' to highlight the lower end explicitly
low_value_spacious = low_value_spacious.sort_values(by='SalePrice').reset_index(drop=True)

# Using df.iloc to select and print the first 5 observations of such low-value houses
low_value_spacious_first_5 = low_value_spacious.iloc[:5, :]

# Print only relevant columns
print(low_value_spacious_first_5[['Neighborhood', 'SalePrice', 'GrLivArea']])
