import pandas as pd

Ames = pd.read_csv('Ames.csv')

# Applying Squared Transformation
Ames['Squared_YearBuilt'] = Ames['YearBuilt'] ** 2
print(f"Skewness after Squared Transformation: {Ames['Squared_YearBuilt'].skew():.5f}")
