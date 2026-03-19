import pandas as pd

Ames = pd.read_csv('Ames.csv')
features = ['LotArea', 'SalePrice', 'TotRmsAbvGrd']

# Define a function to tabulate outliers into a DataFrame
def outliers_dataframes_gaussian(dataframe, features, threshold=3, num_rows=None):
    outliers_dataframes = {}

    for feature in features:
        data = dataframe[feature]
        mean = data.mean()
        std_dev = data.std()
        outliers = data[(data < mean - threshold * std_dev) |
                        (data > mean + threshold * std_dev)]

        # Create a new DataFrame for outliers of the current feature
        outliers_df = dataframe.loc[outliers.index, [feature]].copy()
        outliers_df.rename(columns={feature: 'Outlier Value'}, inplace=True)
        outliers_df['Feature'] = feature
        outliers_df.reset_index(inplace=True)

        # Display specified number of rows (default: full dataframe)
        if num_rows:
            outliers_df = outliers_df.head(num_rows)

        outliers_dataframes[feature] = outliers_df

    return outliers_dataframes

# Example usage with user-defined number of rows = 7
outliers_gaussian_dataframes = outliers_dataframes_gaussian(Ames, features, num_rows=7)

# Print each DataFrame with the original format and capitalized 'index'
for feature, df in outliers_gaussian_dataframes.items():
    df_reset = df.reset_index().rename(columns={'index': 'Index'})
    print(f"Outliers for {feature}:\n", df_reset[['Index', 'Feature', 'Outlier Value']])
    print()
