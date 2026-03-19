import pandas as pd
import scipy.stats

Ames = pd.read_csv('Ames.csv')

# Applying Box-Cox Transformation after checking all values are positive
if (Ames['YearBuilt'] > 0).all():
    Ames['BoxCox_YearBuilt'], _ = scipy.stats.boxcox(Ames['YearBuilt'])
else:
    # Consider alternative transformations or handling strategies
    print("Not all YearBuilt values are positive.")
    print("Consider using Yeo-Johnson or handling negative values.")
print(f"Skewness after Box-Cox Transformation: {Ames['BoxCox_YearBuilt'].skew():.5f}")
