#import pandas
import pandas as pd

# Load the dataset
Ames = pd.read_csv("Ames.csv")

# Calculate the sum of "1stFlrSF", "2ndFlrSF", and "LowQualFinSF"
Ames["CalculatedGrLivArea"] = Ames["1stFlrSF"] + Ames["2ndFlrSF"] + Ames["LowQualFinSF"]

# Compare the calculated sum with the existing "GrLivArea" column
Ames["IsEqual"] = Ames["GrLivArea"] == Ames["CalculatedGrLivArea"]

# Output the percentage of rows where the values match
match_percentage = Ames["IsEqual"].mean() * 100
print(f"Percentage of rows where GrLivArea equals the sum of the other three features: "
      f"{int(match_percentage)}%")
