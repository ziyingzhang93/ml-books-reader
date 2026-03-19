import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
Ames = pd.read_csv('Ames.csv')

# Convert 'LotShape' to a binary feature: 'Regular' and 'Irregular'
Ames['LotShape_Binary'] = \
    Ames['LotShape'].apply(lambda x: 'Regular' if x == 'Reg' else 'Irregular')

# Creating the pair plot, color-coded by 'LotShape_Binary'
sns.pairplot(Ames, vars=['SalePrice', 'OverallQual', 'GrLivArea'], hue='LotShape_Binary',
             palette='Set1', height=2.5, aspect=1.75)

# Display the plot
plt.show()
