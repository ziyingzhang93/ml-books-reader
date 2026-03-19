import pandas as pd
# Importing visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

Ames = pd.read_csv('Ames.csv')

# Setting up the style
sns.set_style("whitegrid")

# Calculate Mean, Median, Mode for SalePrice
mean_saleprice = Ames['SalePrice'].mean()
median_saleprice = Ames['SalePrice'].median()
mode_saleprice = Ames['SalePrice'].mode().values[0]

# Plotting the histogram
plt.figure(figsize=(14, 7))
sns.histplot(x=Ames['SalePrice'], bins=30, kde=True, color="skyblue")
plt.axvline(mean_saleprice, color='r', linestyle='--',
            label=f"Mean: ${mean_saleprice:.2f}")
plt.axvline(median_saleprice, color='g', linestyle='-',
            label=f"Median: ${median_saleprice:.2f}")
plt.axvline(mode_saleprice, color='b', linestyle='-.',
            label=f"Mode: ${mode_saleprice:.2f}")

# Calculating skewness and kurtosis for SalePrice
skewness_saleprice = Ames['SalePrice'].skew()
kurtosis_saleprice = Ames['SalePrice'].kurt()

# Annotations for skewness and kurtosis
text = 'Skewness: {:.2f}\nKurtosis: {:.2f}' \
        .format(Ames['SalePrice'].skew(), Ames['SalePrice'].kurt())
plt.annotate(text, xy=(500000, 100), fontsize=14,
             bbox={"boxstyle": "round,pad=0.3",
                   "edgecolor": "black",
                   "facecolor": "aliceblue"})
plt.title('Histogram of Ames\' Housing Prices with KDE and Reference Lines')
plt.xlabel('Housing Prices')
plt.ylabel('Frequency')
plt.legend()
plt.show()
