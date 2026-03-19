import pandas as pd
import scipy.stats as stats

Ames = pd.read_csv('Ames.csv')

# Perform the ANOVA
f_value, p_value = stats.f_oneway(*[Ames['SalePrice'][Ames['YrSold'] == year]
                                    for year in Ames['YrSold'].unique()])
print(f_value, p_value)
