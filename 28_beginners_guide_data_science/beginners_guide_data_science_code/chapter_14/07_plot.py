import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer

Ames = pd.read_csv('Ames.csv')
Ames['Log_SalePrice'] = np.log(Ames['SalePrice'])
Ames['Sqrt_SalePrice'] = np.sqrt(Ames['SalePrice'])
Ames['BoxCox_SalePrice'], _ = scipy.stats.boxcox(Ames['SalePrice'])
Ames['YeoJohnson_SalePrice'], _ = scipy.stats.yeojohnson(Ames['SalePrice'])
quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
Ames['Quantile_SalePrice'] = \
    quantile_transformer.fit_transform(Ames['SalePrice'].values.reshape(-1, 1)).flatten()

# Plotting the distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Hide unused subplot axes
for ax in axes[6:]:
    ax.axis('off')

# Original SalePrice Distribution
sns.histplot(Ames['SalePrice'], kde=True, bins=30, color='skyblue', ax=axes[0])
axes[0].set_title('Original SalePrice Distribution (Skew: 1.76)')
axes[0].set_xlabel('SalePrice')
axes[0].set_ylabel('Frequency')

# Log Transformed SalePrice
sns.histplot(Ames['Log_SalePrice'], kde=True, bins=30, color='blue', ax=axes[1])
axes[1].set_title('Log Transformed SalePrice (Skew: 0.04172)')
axes[1].set_xlabel('Log of SalePrice')
axes[1].set_ylabel('Frequency')

# Square Root Transformed SalePrice
sns.histplot(Ames['Sqrt_SalePrice'], kde=True, bins=30, color='orange', ax=axes[2])
axes[2].set_title('Square Root Transformed (Skew: 0.90148)')
axes[2].set_xlabel('Square Root of SalePrice')
axes[2].set_ylabel('Frequency')

# Box-Cox Transformed SalePrice
sns.histplot(Ames['BoxCox_SalePrice'], kde=True, bins=30, color='red', ax=axes[3])
axes[3].set_title('Box-Cox Transformed SalePrice (Skew: -0.00436)')
axes[3].set_xlabel('Box-Cox of SalePrice')
axes[3].set_ylabel('Frequency')

# Yeo-Johnson Transformed SalePrice
sns.histplot(Ames['YeoJohnson_SalePrice'], kde=True, bins=30, color='purple', ax=axes[4])
axes[4].set_title('Yeo-Johnson Transformed (Skew: -0.00437)')
axes[4].set_xlabel('Yeo-Johnson of SalePrice')
axes[4].set_ylabel('Frequency')

# Quantile Transformed SalePrice (Normal Distribution)
sns.histplot(Ames['Quantile_SalePrice'], kde=True, bins=30, color='green', ax=axes[5])
axes[5].set_title('Quantile Transformed (Normal Distn, Skew: 0.00286)')
axes[5].set_xlabel('Quantile Transformed SalePrice')
axes[5].set_ylabel('Frequency')

plt.tight_layout(pad=4.0)
plt.show()
