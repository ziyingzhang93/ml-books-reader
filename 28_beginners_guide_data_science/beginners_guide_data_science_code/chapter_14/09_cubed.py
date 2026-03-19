import pandas as pd

Ames = pd.read_csv('Ames.csv')

# Applying Cubed Transformation
Ames['Cubed_YearBuilt'] = Ames['YearBuilt'] ** 3
print(f"Skewness after Cubed Transformation: {Ames['Cubed_YearBuilt'].skew():.5f}")
