# Importing libraries and loading the dataset
import pandas as pd
Ames = pd.read_csv('Ames.csv')

# Descriptive Statistics of Sales Price
sales_price_description = Ames['SalePrice'].describe()
print(sales_price_description)
