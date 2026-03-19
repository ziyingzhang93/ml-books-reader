import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Ames = pd.read_csv('Ames.csv')
specific_houses = Ames.query('BedroomAbvGr > 3 & SalePrice < 300000')

# Visualizing the advanced query results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', hue='BedroomAbvGr',
                data=specific_houses, palette='viridis')
plt.title('Sales Price vs. Ground Living Area')
plt.xlabel('Ground Living Area (sqft)')
plt.ylabel('Sales Price ($)')
plt.legend(title='Bedrooms Above Ground')
plt.show()
