import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Dataset
Ames = pd.read_csv('Ames.csv')

# Calculate the top 10 features most correlated with 'SalePrice'
correlations = Ames.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False)
top_correlations = correlations[1:11]

# Select the top correlated features including SalePrice
selected_features = list(top_correlations.index) + ['SalePrice']

# Compute the correlations for the selected features
correlation_matrix = Ames[selected_features].corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Generate a heatmap
sns.heatmap(correlation_matrix, annot=True,
            cmap="coolwarm", linewidths=.5, fmt=".2f", vmin=-1, vmax=1)

# Title
plt.title("Heatmap of Correlations among Top Features with SalePrice", fontsize=16)

# Show the heatmap
plt.show()
