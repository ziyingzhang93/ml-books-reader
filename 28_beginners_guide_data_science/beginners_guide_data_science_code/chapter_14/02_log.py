import pandas as pd
import numpy as np

Ames = pd.read_csv('Ames.csv')

# Applying Log Transformation
Ames['Log_SalePrice'] = np.log(Ames['SalePrice'])
print(f"Skewness after Log Transformation: {Ames['Log_SalePrice'].skew():.5f}")
