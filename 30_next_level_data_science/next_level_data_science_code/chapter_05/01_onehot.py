# Load only categorical columns without missing values from the Ames dataset
import pandas as pd
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["object"]).dropna(axis=1)
print(f"The shape of the DataFrame before one-hot encoding is: {Ames.shape}")

# Import OneHotEncoder and apply it to Ames:
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
Ames_One_Hot = encoder.fit_transform(Ames)

# Convert the encoded result back to a DataFrame
Ames_encoded_df = pd.DataFrame(Ames_One_Hot,
                               columns=encoder.get_feature_names_out(Ames.columns))

# Display the new DataFrame and it's expanded shape
print(Ames_encoded_df.head())
print(f"The shape of the DataFrame after one-hot encoding is: {Ames_encoded_df.shape}")
