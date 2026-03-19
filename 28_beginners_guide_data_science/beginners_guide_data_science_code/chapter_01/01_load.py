# Load the Ames dataset
import pandas as pd
Ames = pd.read_csv('Ames.csv')

# Dataset shape
print(Ames.shape)

rows, columns = Ames.shape
print(f"The dataset comprises {rows} properties described across {columns} attributes.")
