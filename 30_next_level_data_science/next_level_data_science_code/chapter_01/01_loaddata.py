# Load the Ames dataset
import pandas as pd
Ames = pd.read_csv("Ames.csv")

# Display the first few rows of the dataset and the data type of "SalePrice"
print(Ames.head())

sale_price_dtype = Ames["SalePrice"].dtype
print(f"The data type of 'SalePrice' is {sale_price_dtype}.")
