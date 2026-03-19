import pandas as pd
import numpy as np

Ames = pd.read_csv('Ames.csv')

# Applying Square Root Transformation
Ames['Sqrt_SalePrice'] = np.sqrt(Ames['SalePrice'])
print(f"Skewness after Square Root Transformation: {Ames['Sqrt_SalePrice'].skew():.5f}")
