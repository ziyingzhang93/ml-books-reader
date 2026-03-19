import pandas as pd
import folium

Ames = pd.read_csv('Ames.csv')
Ames['PSF'] = Ames['SalePrice']/Ames['GrLivArea']
value = Ames.query('PSF < 175 & OverallQual == 10 & KitchenQual == "Ex" & Fireplaces >=2')

final_observation_indexes = value.index.tolist()

# Filter the dataset for these observations to get their latitude and longitude
final_locations = Ames.loc[final_observation_indexes, ['Latitude', 'Longitude']]

# Create a Folium map centered around the average location of the final observations
map_center = [final_locations['Latitude'].mean(), final_locations['Longitude'].mean()]
value_map = folium.Map(location=map_center, zoom_start=12)

# Add information to markers
for idx, row in final_locations.iterrows():
    # Extract additional information for the popup
    info = value.loc[idx, ['SalePrice', 'GrLivArea', 'OverallQual', 'KitchenQual',
                           'TotRmsAbvGrd', 'Fireplaces', 'PSF']]
    popup_text = f"""<b>Index:</b> {idx}<br>
                     <b>SalePrice:</b> {info['SalePrice']}<br>
                     <b>GrLivArea:</b> {info['GrLivArea']} sqft<br>
                     <b>OverallQual:</b> {info['OverallQual']}<br>
                     <b>KitchenQual:</b> {info['KitchenQual']}<br>
                     <b>TotRmsAbvGrd:</b> {info['TotRmsAbvGrd']}<br>
                     <b>Fireplaces:</b> {info['Fireplaces']}<br>
                     <b>PSF:</b> ${info['PSF']:.2f} /sqft"""
    folium.Marker([row['Latitude'], row['Longitude']],
                  popup=folium.Popup(popup_text, max_width=250)).add_to(value_map)

# Save the map to an HTML file on working directory
value_map.save('value_map.html')
