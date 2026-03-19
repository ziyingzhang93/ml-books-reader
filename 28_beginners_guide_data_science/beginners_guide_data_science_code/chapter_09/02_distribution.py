import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
Ames = pd.read_csv('Ames.csv')

confidence_level = 0.95
degrees_freedom = Ames['SalePrice'].count() - 1
confidence_interval = stats.t.interval(confidence_level, degrees_freedom,
                                       loc=Ames['SalePrice'].mean(),
                                       scale=Ames['SalePrice'].sem())

# Plot the main histogram
plt.figure(figsize=(10, 7))
plt.hist(Ames['SalePrice'], bins=30, color='lightblue', edgecolor='black', alpha=0.5,
         label='Sales Prices Distribution')

# Vertical lines for sample mean and confidence interval with adjusted styles
plt.axvline(Ames['SalePrice'].mean(), color='blue', linestyle='-',
            label=f'Mean: ${Ames["SalePrice"].mean():,.2f}')
plt.axvline(confidence_interval[0], color='red', linestyle='--',
            label=f'Lower 95% CI: ${confidence_interval[0]:,.2f}')
plt.axvline(confidence_interval[1], color='green', linestyle='--',
            label=f'Upper 95% CI: ${confidence_interval[1]:,.2f}')

# Annotations and labels
plt.title('Distribution of Sales Prices with Confidence Interval', fontsize=20)
plt.xlabel('Sales Price', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.xlim([min(Ames['SalePrice']) - 5000, max(Ames['SalePrice']) + 5000])
plt.legend()
plt.grid(axis='y')
plt.show()
