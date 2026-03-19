import pandas as pd
import matplotlib.colors
import matplotlib.pyplot as plt
import seaborn as sns

Ames = pd.read_csv('Ames.csv')
affordable_houses = Ames.query('SalePrice < 300000 & BedroomAbvGr > 0')
pivot_table = affordable_houses \
              .pivot_table(values='SalePrice', index='Neighborhood',
                           columns='BedroomAbvGr', aggfunc='mean') \
              .round(2) \
              .fillna(0)

# Create a custom color map
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])

# Mask for "zero" values to be colored with a different shade
mask = pivot_table == 0

# Set the size of the plot
plt.figure(figsize=(14, 10))

# Create a heatmap with the mask
sns.heatmap(pivot_table,
            cmap=cmap,
            annot=True,
            fmt=".0f",
            linewidths=.5,
            mask=mask,
            cbar_kws={'label': 'Average Sales Price ($)'})

# Adding title and labels for clarity
plt.title('Average Sales Price by Neighborhood and Number of Bedrooms', fontsize=16)
plt.xlabel('Number of Bedrooms Above Grade', fontsize=12)
plt.ylabel('Neighborhood', fontsize=12)

# Display the heatmap
plt.show()
