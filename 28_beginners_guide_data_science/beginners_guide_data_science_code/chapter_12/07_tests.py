import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro

# Load the dataset
Ames = pd.read_csv('Ames.csv')

# Perform the ANOVA
f_value, p_value = stats.f_oneway(*[Ames['SalePrice'][Ames['YrSold'] == year]
                                    for year in Ames['YrSold'].unique()])
print("F-value:", f_value)
print("p-value:", p_value)

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

# Shapiro-Wilk Test
shapiro_stat, shapiro_p = shapiro(residuals)
print(f"Shapiro-Wilk Test Statistic: {shapiro_stat}")
print(f"P-value: {shapiro_p}")

# Check for equal variances using Levene's test
levene_stat, levene_p = stats.levene(*[Ames['SalePrice'][Ames['YrSold'] == year]
                                      for year in Ames['YrSold'].unique()])
print(f"Levene's Test Statistic: {levene_stat}")
print(f"P-value: {levene_p}")
