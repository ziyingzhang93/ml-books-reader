import pandas as pd
import scipy.stats

Ames = pd.read_csv('Ames.csv')

# Applying Yeo-Johnson Transformation
Ames['YeoJohnson_SalePrice'], _ = scipy.stats.yeojohnson(Ames['SalePrice'])
print("Skewness after Yeo-Johnson Transformation: "
      f"{Ames['YeoJohnson_SalePrice'].skew():.5f}")
