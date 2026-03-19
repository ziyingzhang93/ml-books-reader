import pandas as pd
import folium
from folium.plugins import HeatMap

Ames = pd.read_csv('Ames.csv')

# Filter out rows with NaN values in 'Latitude' or 'Longitude'
Ames_Heat = Ames.dropna(subset=['Latitude', 'Longitude'])

# Group by 'Neighborhood' and calculate mean 'Latitude' and 'Longitude'
neighborhood_locs = Ames_Heat.groupby('Neighborhood') \
                             .agg({'Latitude':'mean', 'Longitude':'mean'}) \
                             .reset_index()

# Create a map centered around Ames, Iowa
ames_map_center = [Ames_Heat['Latitude'].mean(), Ames_Heat['Longitude'].mean()]
ames_heatmap = folium.Map(location=ames_map_center, zoom_start=12)

# Extract latitude and longitude data for the heatmap
heat_data = [(lat,lon) for lat, lon in zip(Ames_Heat['Latitude'], Ames_Heat['Longitude'])]

# Create and add a HeatMap layer to the map
HeatMap(heat_data, radius=12).add_to(ames_heatmap)

# Add one black flag per neighborhood to the map
for index, row in neighborhood_locs.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=row['Neighborhood'],
        icon=folium.Icon(color='black', icon='flag')
    ).add_to(ames_heatmap)

# Save the map to an HTML file in the working directory
ames_heatmap.save('ames_heatmap.html')
