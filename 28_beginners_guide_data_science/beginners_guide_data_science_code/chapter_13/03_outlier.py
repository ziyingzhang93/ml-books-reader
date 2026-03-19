import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Ames = pd.read_csv('Ames.csv')
feature_names_full = {
    'LotArea': 'Lot Area (sq ft)',
    'SalePrice': 'Sales Price (US$)',
    'TotRmsAbvGrd': 'Total Rooms Above Ground'
}
features = ['LotArea', 'SalePrice', 'TotRmsAbvGrd']

# Define a function to detect outliers using the Gaussian model
def detect_outliers_gaussian(dataframe, features, threshold=3):
    outliers_summary = {}

    for feature in features:
        data = dataframe[feature]
        mean = data.mean()
        std_dev = data.std()
        outliers = data[(data < mean - threshold * std_dev) |
                        (data > mean + threshold * std_dev)]
        outliers_summary[feature] = len(outliers)

        # Visualization
        plt.figure(figsize=(12, 6))
        sns.histplot(data, color="lightblue")
        plt.axvline(mean, color='r', linestyle='-', label=f'Mean: {mean:.2f}')
        plt.axvline(mean - threshold * std_dev, color='y', linestyle='--',
                    label=f'—{threshold} std devs')
        plt.axvline(mean + threshold * std_dev, color='g', linestyle='--',
                    label=f'+{threshold} std devs')

        # Annotate upper 3rd std dev value
        annotate_text = f'{mean + threshold * std_dev:.2f}'
        plt.annotate(annotate_text, xy=(mean + threshold * std_dev, 0),
                     xytext=(mean + (threshold + 1.45) * std_dev, 50),
                     arrowprops={'facecolor': 'black',
                                 'arrowstyle': 'wedge,tail_width=0.7'},
                     fontsize=12, ha='center')

        plt.title(f'Distribution of {feature_names_full[feature]} with Outliers',
                  fontsize=16)
        plt.xlabel(feature_names_full[feature], fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.legend()
        plt.show()

    return outliers_summary

outliers_gaussian_summary = detect_outliers_gaussian(Ames, features)
print(outliers_gaussian_summary)
