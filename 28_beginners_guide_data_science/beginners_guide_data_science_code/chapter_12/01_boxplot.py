import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
Ames = pd.read_csv('Ames.csv')

# Convert 'YrSold' to a categorical variable
Ames['YrSold'] = Ames['YrSold'].astype('category')

plt.figure(figsize=(10, 6))
sns.boxplot(x=Ames['YrSold'], y=Ames['SalePrice'], hue=Ames['YrSold'])
plt.title('Boxplot of Sales Prices by Year', fontsize=18)
plt.xlabel('Year Sold', fontsize=15)
plt.ylabel('Sales Price (US$)', fontsize=15)
plt.legend('')
plt.show()
