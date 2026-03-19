import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
Ames = pd.read_csv('Ames.csv')

# Calculate the correlation of all features with 'SalePrice'
correlations = Ames.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False)

# Top 5 features most correlated with 'SalePrice' (excluding 'SalePrice' itself)
top_5_features = correlations.index[1:6]

# Creating the pair plot for these features and 'SalePrice'
# Adjust the size by setting height and aspect
sns.pairplot(Ames, vars=['SalePrice'] + list(top_5_features), height=1.35, aspect=1.85)

# Displaying the plot
plt.show()
