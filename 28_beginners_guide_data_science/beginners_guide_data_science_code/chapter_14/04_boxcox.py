import pandas as pd
import scipy.stats

Ames = pd.read_csv('Ames.csv')

# Applying Box-Cox Transformation after checking all values are positive
if (Ames['SalePrice'] > 0).all():
    Ames['BoxCox_SalePrice'], lmbda = scipy.stats.boxcox(Ames['SalePrice'])
else:
    # Consider alternative transformations or handling strategies
    print("Not all SalePrice values are positive.")
    print("Consider using Yeo-Johnson or handling negative values.")
print(f"Skewness after Box-Cox Transformation: {Ames['BoxCox_SalePrice'].skew():.5f}")
