import pandas as pd
import scipy.stats

Ames = pd.read_csv('Ames.csv')

# Applying Yeo-Johnson Transformation
Ames['YeoJohnson_YearBuilt'], _ = scipy.stats.yeojohnson(Ames['YearBuilt'])
print("Skewness after Yeo-Johnson Transformation: "
      f"{Ames['YeoJohnson_YearBuilt'].skew():.5f}")
