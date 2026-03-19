import pandas as pd
import numpy as np

# Create a DataFrame with various types of missing values
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': ['a', 'b', None, 'd', 'e'],
    'C': [np.nan, np.nan, np.nan, np.nan, np.nan],
    'D': [1, 2, 3, 4, 5]
})

# Use isnull() to identify missing values
missing_data = df.isnull().sum()

print(df)
print()
print(missing_data)
