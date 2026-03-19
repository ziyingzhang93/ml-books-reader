import pandas as pd
from scipy.stats import chi2_contingency

# Load the dataset
Ames = pd.read_csv('Ames.csv')

# Extracting the relevant columns
data = Ames[['ExterQual', 'GarageFinish']].copy()

# Filling missing values in the 'GarageFinish' column with 'No Garage'
data['GarageFinish'] = data['GarageFinish'].fillna('No Garage')

# Grouping 'GarageFinish' into 'With Garage' and 'No Garage'
data['Garage Group'] \
    = data['GarageFinish'] \
      .apply(lambda x: 'With Garage' if x != 'No Garage' else 'No Garage')

# Grouping 'ExterQual' into 'Great' and 'Average'
data['Quality Group'] \
    = data['ExterQual'].apply(lambda x: 'Great' if x in ['Ex', 'Gd'] else 'Average')

# Constructing the simplified contingency table
simplified_contingency_table = pd.crosstab(data['Quality Group'], data['Garage Group'])

#Printing the Observed Frequency
print("Observed Frequencies:")
observed_df = pd.DataFrame(simplified_contingency_table,
                           index=["Average", "Great"],
                           columns=["No Garage", "With Garage"])
print(observed_df)
print()

# Performing the chi-squared test
chi2_stat, p_value, _, expected_freq = chi2_contingency(simplified_contingency_table)

# Printing the Expected Frequencies
print("Expected Frequencies:")
print(pd.DataFrame(expected_freq,
                   index=["Average", "Great"],
                   columns=["No Garage", "With Garage"]).round(1))
print()

# Printing the results of the test
print(f"Chi-squared Statistic: {chi2_stat:.4f}")
print(f"p-value: {p_value:.4e}")
