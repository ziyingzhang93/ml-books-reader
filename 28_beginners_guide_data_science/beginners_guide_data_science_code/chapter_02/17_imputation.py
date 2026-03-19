# Load the Ames dataset
import pandas as pd
import numpy as np
Ames = pd.read_csv('Ames.csv')

# Calculating the percentage of missing values for each column
missing_data = Ames.isnull().sum()
missing_percentage = (missing_data / len(Ames)) * 100
data_type = Ames.dtypes

# Combining the counts and percentages into a DataFrame for better visualization
missing_info = pd.DataFrame({'Missing Values': missing_data,
                             'Percentage': missing_percentage,
                             'Data Type': data_type})

# Sorting the DataFrame by the percentage of missing values in descending order
missing_info = missing_info.sort_values(by='Percentage', ascending=False)

# Display columns with missing values of numeric data type
print(missing_info[(missing_info['Missing Values'] > 0)
                   & (missing_info['Data Type'] == np.number)])

# Initialize a DataFrame to store the concise information
concise_info = pd.DataFrame(columns=['Feature',
                                     'Missing Values After Imputation',
                                     'Mean Value Used to Impute'])

# Identify and impute missing numerical values, and store the related concise information
missing_numeric_df = missing_info[(missing_info['Missing Values'] > 0) &
                                  (missing_info['Data Type'] == np.number)]

for item in missing_numeric_df.index.tolist():
    mean_value = Ames[item].mean(skipna=True)
    Ames[item].fillna(mean_value, inplace=True)

    # Append the concise information to the concise_info DataFrame
    concise_info.loc[len(concise_info)] = pd.Series({
        'Feature': item,
        'Missing Values After Imputation': Ames[item].isnull().sum(),
        # This should be 0 as you are imputing all missing values
        'Mean Value Used to Impute': mean_value
    })

# Display the concise_info DataFrame
print(concise_info)

missing_values_count = Ames.isnull().sum().sum()
print(f'The DataFrame has a total of {missing_values_count} missing values.')
