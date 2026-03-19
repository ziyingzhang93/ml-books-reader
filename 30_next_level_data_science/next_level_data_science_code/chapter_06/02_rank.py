import pandas as pd

Ames = pd.read_csv("Ames.csv")
neighbor_stats = Ames.groupby("Neighborhood")["SalePrice"] \
                     .agg(["count", "mean"]) \
                     .sort_values(by="mean")
print(neighbor_stats.round(0).astype(int))
