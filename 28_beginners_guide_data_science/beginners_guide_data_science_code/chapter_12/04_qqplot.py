import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

Ames = pd.read_csv('Ames.csv')

# Fit an ordinary least squares model and get residuals
model = sm.OLS(Ames['SalePrice'], Ames['YrSold'].astype('int')).fit()
residuals = model.resid

# Plot QQ plot
sm.qqplot(residuals, line='s')
plt.title('Normality Assessment of Residuals via QQ Plot', fontsize=18)
plt.xlabel('Theoretical Quantiles', fontsize=15)
plt.ylabel('Sample Residual Quantiles', fontsize=15)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
