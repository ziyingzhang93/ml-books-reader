import pandas as pd

# Load the dataset
Ames = pd.read_csv('Ames.csv')

# Define the quartiles
quantiles = Ames['SalePrice'].quantile([0.25, 0.5, 0.75])

# Function to categorize each row
def categorize_by_price(row):
    if row['SalePrice'] <= quantiles.iloc[0]:
        return 'Low'
    elif row['SalePrice'] <= quantiles.iloc[1]:
        return 'Medium'
    elif row['SalePrice'] <= quantiles.iloc[2]:
        return 'High'
    else:
        return 'Premium'

# Apply the function to create a new column
Ames['Price_Category'] = Ames.apply(categorize_by_price, axis=1)
print(Ames[['SalePrice','Price_Category']])
