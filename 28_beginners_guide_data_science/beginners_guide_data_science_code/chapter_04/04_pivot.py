import pandas as pd

def categorize_by_price(row):
    if row['SalePrice'] <= quantiles.iloc[0]:
        return 'Low'
    elif row['SalePrice'] <= quantiles.iloc[1]:
        return 'Medium'
    elif row['SalePrice'] <= quantiles.iloc[2]:
        return 'High'
    else:
        return 'Premium'

Ames = pd.read_csv('Ames.csv')
quantiles = Ames['SalePrice'].quantile([0.25, 0.5, 0.75])
Ames['Price_Category'] = Ames.apply(categorize_by_price, axis=1)

# Split original dataset into 4 DataFrames by Price Category
low_priced_homes = Ames.query('Price_Category == "Low"')
medium_priced_homes = Ames.query('Price_Category == "Medium"')
high_priced_homes = Ames.query('Price_Category == "High"')
premium_priced_homes = Ames.query('Price_Category == "Premium"')

# Stacking Low and Medium categories into an "affordable_homes" DataFrame
affordable_homes = pd.concat([low_priced_homes, medium_priced_homes])

# Stacking High and Premium categories into a "luxury_homes" DataFrame
luxury_homes = pd.concat([high_priced_homes, premium_priced_homes])

# Creating pivot tables with both mean living area and home count
aggfunc = {'GrLivArea': 'mean', 'Fireplaces': 'count'}
pivot_affordable = affordable_homes.pivot_table(index='Fireplaces', aggfunc=aggfunc)
pivot_luxury = luxury_homes.pivot_table(index='Fireplaces', aggfunc=aggfunc)

# Renaming columns and index labels separately
rename_rules = {'GrLivArea': 'AvLivArea', 'Fireplaces': 'HmCount'}

pivot_affordable.rename(columns=rename_rules, inplace=True)
pivot_affordable.index.name = 'Fire'

pivot_luxury.rename(columns=rename_rules, inplace=True)
pivot_luxury.index.name = 'Fire'

# View the pivot tables
print(pivot_affordable)
print(pivot_luxury)
