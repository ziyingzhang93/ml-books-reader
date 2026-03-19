import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

Ames = pd.read_csv('Ames.csv')

# Perform the Kruskal-Wallis H-test
H_statistic, kruskal_p_value = stats.kruskal(*[Ames['SalePrice'][Ames['YrSold'] == year]
                                               for year in Ames['YrSold'].unique()])
print(H_statistic, kruskal_p_value)

# Plot histograms of Sales Price for each year
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 8), sharex=True)

for idx, year in enumerate(sorted(Ames['YrSold'].unique())):
    sns.histplot(Ames[Ames['YrSold'] == year]['SalePrice'], kde=True, ax=axes[idx],
                 color='skyblue')
    axes[idx].set_title(f'Distribution of Sales Prices for Year {year}', fontsize=16)
    axes[idx].set_ylabel('Frequency', fontsize=14)
    if idx == 4:
        axes[idx].set_xlabel('Sales Price', fontsize=15)
    else:
        axes[idx].set_xlabel('')

plt.tight_layout()
plt.show()

# Run KS Test from scipy.stats
results = {}
for i, year1 in enumerate(sorted(Ames['YrSold'].unique())):
    for j, year2 in enumerate(sorted(Ames['YrSold'].unique())):
        if i < j:
            ks_stat, ks_p = ks_2samp(Ames[Ames['YrSold'] == year1]['SalePrice'],
                                     Ames[Ames['YrSold'] == year2]['SalePrice'])
            results[f"{year1} vs {year2}"] = (ks_stat, ks_p)

# Convert the results into a DataFrame for tabular representation
ks_df = pd.DataFrame(results).transpose()
ks_df.columns = ['KS Statistic', 'P-value']
ks_df.reset_index(inplace=True)
ks_df.rename(columns={'index': 'Years Compared'}, inplace=True)
print(ks_df)
