import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
Ames = pd.read_csv('Ames.csv')

# Horizontal box plot with annotations
plt.figure(figsize=(12, 8))

# Plotting the box plot with specified color and style
sns.boxplot(x=Ames['SalePrice'], color='skyblue', showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "red",
                       "markeredgecolor": "red", "markersize":10})

# Plotting arrows for Q1, Median and Q3
q1_saleprice = Ames['SalePrice'].quantile(0.25)
q2_saleprice = Ames['SalePrice'].quantile(0.50)
q3_saleprice = Ames['SalePrice'].quantile(0.75)
plt.annotate('Q1', xy=(q1_saleprice, 0.30), xytext=(q1_saleprice - 70000, 0.45),
             arrowprops={'edgecolor': 'black', 'arrowstyle': '->'}, fontsize=14)
plt.annotate('Q3', xy=(q3_saleprice, 0.30), xytext=(q3_saleprice + 20000, 0.45),
             arrowprops={'edgecolor': 'black', 'arrowstyle': '->'}, fontsize=14)
plt.annotate('Median', xy=(q2_saleprice, 0.20), xytext=(q2_saleprice - 90000, 0.05),
             arrowprops={'edgecolor': 'black', 'arrowstyle': '->'}, fontsize=14)

# Titles, labels, and legends
plt.title('Box Plot Ames\' Housing Prices', fontsize=16)
plt.xlabel('Housing Prices', fontsize=14)
plt.yticks([])  # Hide y-axis tick labels
plt.legend(handles=[Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
           markersize=10, label='Mean')], loc='upper left', fontsize=14)

plt.tight_layout()
plt.show()
