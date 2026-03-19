import pandas as pd
Ames = pd.read_csv('Ames.csv')

# Calculating mean and median sales price by year
summary_table = Ames.groupby('YrSold')['SalePrice'].agg(['mean', 'median'])

# Rounding the values for better presentation
summary_table = summary_table.round(2)
print(summary_table)
