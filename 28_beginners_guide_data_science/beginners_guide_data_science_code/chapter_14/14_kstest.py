import pandas as pd
import numpy as np
import scipy.stats
from sklearn.preprocessing import QuantileTransformer

Ames = pd.read_csv('Ames.csv')
Ames['Log_SalePrice'] = np.log(Ames['SalePrice'])
Ames['Sqrt_SalePrice'] = np.sqrt(Ames['SalePrice'])
Ames['BoxCox_SalePrice'], _ = scipy.stats.boxcox(Ames['SalePrice'])
Ames['YeoJohnson_SalePrice'], _ = scipy.stats.yeojohnson(Ames['SalePrice'])
quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
Ames['Quantile_SalePrice'] = \
    quantile_transformer.fit_transform(Ames['SalePrice'].values.reshape(-1, 1)).flatten()
Ames['Squared_YearBuilt'] = Ames['YearBuilt'] ** 2
Ames['Cubed_YearBuilt'] = Ames['YearBuilt'] ** 3
Ames['BoxCox_YearBuilt'], _ = scipy.stats.boxcox(Ames['YearBuilt'])
Ames['YeoJohnson_YearBuilt'], _ = scipy.stats.yeojohnson(Ames['YearBuilt'])
Ames['Quantile_YearBuilt'] = \
    quantile_transformer.fit_transform(Ames['YearBuilt'].values.reshape(-1, 1)).flatten()

# Run the Kolmogorov-Smirnov tests for the 10 cases
transformations = ["Log_SalePrice", "Sqrt_SalePrice", "BoxCox_SalePrice",
                    "YeoJohnson_SalePrice", "Quantile_SalePrice",
                    "Squared_YearBuilt", "Cubed_YearBuilt", "BoxCox_YearBuilt",
                    "YeoJohnson_YearBuilt", "Quantile_YearBuilt"]

# Standardizing the transformations before performing KS test
ks_test_results = {}
for transformation in transformations:
    standardized_data = \
        (Ames[transformation] - Ames[transformation].mean()) / Ames[transformation].std()
    ks_stat, ks_p_value = scipy.stats.kstest(standardized_data, 'norm')
    ks_test_results[transformation] = (ks_stat, ks_p_value)

# Convert results to DataFrame for easier comparison
ks_test_results_df = pd.DataFrame.from_dict(ks_test_results, orient='index',
                                            columns=['KS Statistic', 'P-Value'])
print(ks_test_results_df.round(5))
