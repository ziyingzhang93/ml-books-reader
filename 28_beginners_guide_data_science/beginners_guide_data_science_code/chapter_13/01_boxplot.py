import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Ames = pd.read_csv('Ames.csv')

# Define feature names in full form for titles and axis
feature_names_full = {
    'LotArea': 'Lot Area (sq ft)',
    'SalePrice': 'Sales Price (US$)',
    'TotRmsAbvGrd': 'Total Rooms Above Ground'
}

plt.figure(figsize=(18, 6))
features = ['LotArea', 'SalePrice', 'TotRmsAbvGrd']

for i, feature in enumerate(features, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=Ames[feature], color="lightblue")
    plt.title(feature_names_full[feature], fontsize=16)
    plt.ylabel(feature_names_full[feature], fontsize=14)
    plt.xlabel('')  # Removing the x-axis label as it's not needed

plt.tight_layout()
plt.show()
