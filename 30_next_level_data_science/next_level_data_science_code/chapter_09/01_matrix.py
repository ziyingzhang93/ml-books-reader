# Import necessary libraries to check and compare number of columns vs rank of dataset
import pandas as pd
import numpy as np

# Load the dataset
Ames = pd.read_csv("Ames.csv")

# Select numerical columns without missing values
numerical_data = Ames.select_dtypes(include=[np.number]).dropna(axis=1)

# Calculate the matrix rank
rank = np.linalg.matrix_rank(numerical_data.values)

# Number of features
num_features = numerical_data.shape[1]

# Print the rank and the number of features
print(f"Numerical features without missing values: {num_features}")
print(f"Rank: {rank}")
