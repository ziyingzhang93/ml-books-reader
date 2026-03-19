import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Ames = pd.read_csv('Ames.csv')
specific_houses = Ames.query('BedroomAbvGr > 3 & SalePrice < 300000')
grouped_data = specific_houses.groupby('Neighborhood').agg({
    'SalePrice': ['mean', 'count']
})
grouped_data.columns = ['Average Sales Price', 'House Count']
grouped_data['Average Sales Price'] = grouped_data['Average Sales Price'].round(2)

# 'Neighborhood' was index, reset to make it a column then sort by price
grouped_data_reset = grouped_data.reset_index().sort_values(by='Average Sales Price')

# Set the aesthetic style of the plots
sns.set_theme(style="whitegrid")

# Create the bar plot
plt.figure(figsize=(12, 8))
barplot = sns.barplot(
    x='Neighborhood',
    y='Average Sales Price',
    data=grouped_data_reset,
    palette="coolwarm",
    hue='Neighborhood',
    legend=False,
    errorbar=None  # Removes the confidence interval bars
)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Annotate each bar with the house count, using enumerate to get the index for positioning
for index, value in enumerate(grouped_data_reset['Average Sales Price']):
    house_count = grouped_data_reset.iloc[index]['House Count']
    plt.text(index, value, f'{house_count}', ha='center', va='bottom')

plt.title('Average Sales Price by Neighborhood', fontsize=18)
plt.xlabel('Neighborhood')
plt.ylabel('Average Sales Price ($)')

plt.tight_layout()  # Adjust the layout
plt.show()
