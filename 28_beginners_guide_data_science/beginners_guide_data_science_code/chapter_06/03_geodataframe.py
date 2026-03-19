import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Load the dataset
Ames = pd.read_csv('Ames.csv')

# Convert the DataFrame to a GeoDataFrame
geometry = [Point(xy) for xy in zip(Ames['Longitude'], Ames['Latitude'])]
geo_df = gpd.GeoDataFrame(Ames, geometry=geometry)
print(geo_df)
