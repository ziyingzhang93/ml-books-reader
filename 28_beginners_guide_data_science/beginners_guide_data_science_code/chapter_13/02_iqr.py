import pandas as pd

Ames = pd.read_csv('Ames.csv')
features = ['LotArea', 'SalePrice', 'TotRmsAbvGrd']

def detect_outliers_iqr_summary(dataframe, features):
    outliers_summary = {}

    for feature in features:
        data = dataframe[feature]
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outliers_summary[feature] = len(outliers)

    return outliers_summary

outliers_summary = detect_outliers_iqr_summary(Ames, features)
print(outliers_summary)
