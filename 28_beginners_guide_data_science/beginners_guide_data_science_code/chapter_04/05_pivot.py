import pandas as pd

def categorize_by_price(row):
    if row['SalePrice'] <= quantiles.iloc[0]:
        return 'Low'
    elif row['SalePrice'] <= quantiles.iloc[1]:
        return 'Medium'
    elif row['SalePrice'] <= quantiles.iloc[2]:
        return 'High'
    else:
        return 'Premium'

Ames = pd.read_csv('Ames.csv')
quantiles = Ames['SalePrice'].quantile([0.25, 0.5, 0.75])
Ames['Price_Category'] = Ames.apply(categorize_by_price, axis=1)

pivot = Ames.pivot_table(index="Fireplaces",
                         columns="Price_Category",
                         aggfunc={'GrLivArea':'mean', 'Fireplaces':'count'})
print(pivot)
