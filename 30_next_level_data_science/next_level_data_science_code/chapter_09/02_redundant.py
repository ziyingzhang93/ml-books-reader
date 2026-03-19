# Creating and using a function to identify redundant features in a dataset
import pandas as pd
import numpy as np

def find_redundant_features(data):
    """
    Identifies and returns redundant features in a dataset based on matrix rank.
    A feature is considered redundant if removing it does not decrease the rank of the
    dataset, indicating that it can be expressed as a linear combination of other
    features.

    Parameters:
        data (DataFrame): The numerical dataset to analyze.

    Returns:
        list: A list of redundant feature names.
    """

    # Calculate the matrix rank of the original dataset
    original_rank = np.linalg.matrix_rank(data)
    redundant_features = []

    for column in data.columns:
        # Create a new dataset without this column
        temp_data = data.drop(column, axis=1)
        # Calculate the rank of the new dataset
        temp_rank = np.linalg.matrix_rank(temp_data)

        # If the rank does not decrease, the removed column is redundant
        if temp_rank == original_rank:
            redundant_features.append(column)

    return redundant_features

# Usage of the function with the numerical data
Ames = pd.read_csv("Ames.csv")
numerical_data = Ames.select_dtypes(include=[np.number]).dropna(axis=1)
redundant_features = find_redundant_features(numerical_data)
print("Redundant features:", redundant_features)
