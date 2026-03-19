import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
Ames = pd.read_csv('Ames.csv')

# Calculate skewness
sale_price_skew = Ames['SalePrice'].skew()
year_built_skew = Ames['YearBuilt'].skew()

# Set the style of seaborn
sns.set(style='whitegrid')

# Create a figure for 2 subplots (1 row, 2 columns)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot for SalePrice (positively skewed)
sns.histplot(Ames['SalePrice'], kde=True, ax=ax[0], color='skyblue')
ax[0].set_title('Distribution of SalePrice (Positive Skew)', fontsize=16)
ax[0].set_xlabel('SalePrice')
ax[0].set_ylabel('Frequency')

# Annotate Skewness
ax[0].text(0.5, 0.5, f'Skew: {sale_price_skew:.2f}', transform=ax[0].transAxes,
           horizontalalignment='right', color='black', weight='bold',
           fontsize=14)

# Plot for YearBuilt (negatively skewed)
sns.histplot(Ames['YearBuilt'], kde=True, ax=ax[1], color='salmon')
ax[1].set_title('Distribution of YearBuilt (Negative Skew)', fontsize=16)
ax[1].set_xlabel('YearBuilt')
ax[1].set_ylabel('Frequency')

# Annotate Skewness
ax[1].text(0.5, 0.5, f'Skew: {year_built_skew:.2f}', transform=ax[1].transAxes,
           horizontalalignment='right', color='black', weight='bold',
           fontsize=14)

plt.tight_layout()
plt.show()
