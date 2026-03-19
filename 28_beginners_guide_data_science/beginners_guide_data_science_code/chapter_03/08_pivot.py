import pandas as pd
Ames = pd.read_csv('Ames.csv')

# Filter for houses priced below $300,000 and with at least 1 bedroom above grade
affordable_houses = Ames.query('SalePrice < 300000 & BedroomAbvGr > 0')

# Create pivot table to analyze average sale price by neighborhood and number of bedrooms
pivot_table = affordable_houses.pivot_table(values='SalePrice',
                                            index='Neighborhood',
                                            columns='BedroomAbvGr',
                                            aggfunc='mean').round(2)

# Fill missing values (combination not exist) with 0 to avoid seeing NaN
pivot_table = pivot_table.fillna(0)

# Adjust pandas display options to ensure all columns are shown
pd.set_option('display.max_columns', None)
print(pivot_table)
