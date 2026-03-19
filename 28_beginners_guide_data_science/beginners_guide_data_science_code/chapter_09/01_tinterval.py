import scipy.stats as stats
import pandas as pd
Ames = pd.read_csv('Ames.csv')

#Define the confidence level and degrees of freedom
confidence_level = 0.95
degrees_freedom = Ames['SalePrice'].count() - 1

#Calculate the confidence interval for 'SalePrice'
confidence_interval = stats.t.interval(confidence_level, degrees_freedom,
                                       loc=Ames['SalePrice'].mean(),
                                       scale=Ames['SalePrice'].sem())

# Print out the sentence with the confidence interval figures
print(f"The 95% confidence interval for the true mean sales price of all houses in Ames "
      f"is between ${confidence_interval[0]:.2f} and ${confidence_interval[1]:.2f}.")
