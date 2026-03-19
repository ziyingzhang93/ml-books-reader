import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer

Ames = pd.read_csv('Ames.csv')
Ames['Squared_YearBuilt'] = Ames['YearBuilt'] ** 2
Ames['Cubed_YearBuilt'] = Ames['YearBuilt'] ** 3
Ames['BoxCox_YearBuilt'], _ = scipy.stats.boxcox(Ames['YearBuilt'])
Ames['YeoJohnson_YearBuilt'], _ = scipy.stats.yeojohnson(Ames['YearBuilt'])
quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
Ames['Quantile_YearBuilt'] = \
    quantile_transformer.fit_transform(Ames['YearBuilt'].values.reshape(-1, 1)).flatten()

# Plotting the distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Original YearBuilt Distribution
sns.histplot(Ames['YearBuilt'], kde=True, bins=30, color='skyblue', ax=axes[0])
axes[0].set_title(f'Original YearBuilt Distr. (Skew: {Ames["YearBuilt"].skew():.5f})')
axes[0].set_xlabel('YearBuilt')
axes[0].set_ylabel('Frequency')

# Squared YearBuilt
sns.histplot(Ames['Squared_YearBuilt'], kde=True, bins=30, color='blue', ax=axes[1])
axes[1].set_title(f'Squared YearBuilt (Skew: {Ames["Squared_YearBuilt"].skew():.5f})')
axes[1].set_xlabel('Squared YearBuilt')
axes[1].set_ylabel('Frequency')

# Cubed YearBuilt
sns.histplot(Ames['Cubed_YearBuilt'], kde=True, bins=30, color='orange', ax=axes[2])
axes[2].set_title(f'Cubed YearBuilt (Skew: {Ames["Cubed_YearBuilt"].skew():.5f})')
axes[2].set_xlabel('Cubed YearBuilt')
axes[2].set_ylabel('Frequency')

# Box-Cox Transformed YearBuilt
sns.histplot(Ames['BoxCox_YearBuilt'], kde=True, bins=30, color='red', ax=axes[3])
axes[3].set_title(f'Box-Cox Transformed (Skew: {Ames["BoxCox_YearBuilt"].skew():.5f})')
axes[3].set_xlabel('Box-Cox YearBuilt')
axes[3].set_ylabel('Frequency')

# Yeo-Johnson Transformed YearBuilt
sns.histplot(Ames['YeoJohnson_YearBuilt'], kde=True, bins=30, color='purple', ax=axes[4])
axes[4].set_title('Yeo-Johnson Transformed (Skew: '
                  f'{Ames["YeoJohnson_YearBuilt"].skew():.5f})')
axes[4].set_xlabel('Yeo-Johnson YearBuilt')
axes[4].set_ylabel('Frequency')

# Quantile Transformed YearBuilt (Normal Distribution)
sns.histplot(Ames['Quantile_YearBuilt'], kde=True, bins=30, color='green', ax=axes[5])
axes[5].set_title('Quantile Transformed (Normal Dist., '
                  f'Skew: {Ames["Quantile_YearBuilt"].skew():.5f})')
axes[5].set_xlabel('Quantile Transformed YearBuilt')
axes[5].set_ylabel('Frequency')

plt.tight_layout(pad=4.0)
plt.show()
