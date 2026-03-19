import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Ames = pd.read_csv('Ames.csv')
missing_data = Ames.isnull().sum()
missing_percentage = (missing_data / len(Ames)) * 100
missing_info = pd.DataFrame({'Missing Values': missing_data,
                             'Percentage': missing_percentage})

# Filter to show only the top 15 columns with the most missing values
top_15_missing_info = missing_info.nlargest(15, 'Percentage').reset_index()
print(top_15_missing_info)

# Create the horizontal bar plot using seaborn
plt.figure(figsize=(12, 8))
sns.barplot(x='Percentage', y="index", hue="index", data=top_15_missing_info, orient='h')
plt.title('Top 15 Features with Missing Percentages', fontsize=20)
plt.xlabel('Percentage of Missing Values', fontsize=16)
plt.ylabel('Features', fontsize=16)
#plt.yticks(fontsize=11)
plt.show()
