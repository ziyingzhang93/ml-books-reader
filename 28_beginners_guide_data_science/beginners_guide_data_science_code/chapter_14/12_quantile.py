import pandas as pd
from sklearn.preprocessing import QuantileTransformer

Ames = pd.read_csv('Ames.csv')

# Applying Quantile Transformation to follow a normal distribution
quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
Ames['Quantile_YearBuilt'] = \
    quantile_transformer.fit_transform(Ames['YearBuilt'].values.reshape(-1, 1)).flatten()
print(f"Skewness after Quantile Transformation: {Ames['Quantile_YearBuilt'].skew():.5f}")
