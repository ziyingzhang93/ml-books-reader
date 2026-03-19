# Load the Dataset
import pandas as pd
Ames = pd.read_csv('Ames.csv')

# Calculate the correlation of all features with 'SalePrice'
# Set numeric_only=True to limit the output to numeric columns
correlations = Ames.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False)

# Display the top 10 features most correlated with 'SalePrice'
top_correlations = correlations[1:11]
print(top_correlations)
