import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Ames = pd.read_csv('Ames.csv')

# Setting up the figure and axes
fig, ax = plt.subplots(2, 2, figsize=(15, 12))

# Scatter plot for SalePrice vs. OverallQual
sns.scatterplot(x=Ames['OverallQual'], y=Ames['SalePrice'], ax=ax[0, 0],
                color='blue', alpha=0.6)
ax[0, 0].set_title('House Prices vs. Overall Quality')
ax[0, 0].set_ylabel('House Prices')
ax[0, 0].set_xlabel('Overall Quality')

# Scatter plot for SalePrice vs. GrLivArea
sns.scatterplot(x=Ames['GrLivArea'], y=Ames['SalePrice'], ax=ax[0, 1],
                color='red', alpha=0.6)
ax[0, 1].set_title('House Prices vs. Ground Living Area')
ax[0, 1].set_ylabel('House Prices')
ax[0, 1].set_xlabel('Above Ground Living Area (sq. ft.)')

# Scatter plot for SalePrice vs. TotalBsmtSF
sns.scatterplot(x=Ames['TotalBsmtSF'], y=Ames['SalePrice'], ax=ax[1, 0],
                color='green', alpha=0.6)
ax[1, 0].set_title('House Prices vs. Total Basement Area')
ax[1, 0].set_ylabel('House Prices')
ax[1, 0].set_xlabel('Total Basement Area (sq. ft.)')

# Scatter plot for SalePrice vs. 1stFlrSF
sns.scatterplot(x=Ames['1stFlrSF'], y=Ames['SalePrice'], ax=ax[1, 1],
                color='purple', alpha=0.6)
ax[1, 1].set_title('House Prices vs. First Floor Area')
ax[1, 1].set_ylabel('House Prices')
ax[1, 1].set_xlabel('First Floor Area (sq. ft.)')

# Adjust layout
plt.tight_layout(pad=3.0)
plt.show()
