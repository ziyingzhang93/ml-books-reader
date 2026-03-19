import pandas as pd

Ames = pd.read_csv('Ames.csv')

# Advanced querying: Select houses with more than 3 bedrooms and priced below $300,000
specific_houses = Ames.query('BedroomAbvGr > 3 & SalePrice < 300000')

# Group by neighborhood, then calculate the average and total price, and count the houses
grouped_data = specific_houses.groupby('Neighborhood').agg({
    'SalePrice': ['mean', 'count']
})

# 'Neighborhood' is the index but you should rename the columns for clarity
grouped_data.columns = ['Average Sales Price', 'House Count']

# Round the average sale price to 2 decimal places
grouped_data['Averages Sales Price'] = grouped_data['Average Sales Price'].round(2)

print(grouped_data)
