# Required libraries
import pandas as pd
import folium
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load the station data generated previously
df = pd.read_excel("fresno_to_cheshire_stations.xlsx")

# Create the map centered around the midpoint of the route
map_center = [39.5, -95.5]  # roughly central USA
m = folium.Map(location=map_center, zoom_start=5)

# Load route geometry from OpenRouteService again
cities_coords = {
    'Fresno, CA': (-119.7871, 36.7378),
    'Tulsa, OK': (-95.9928, 36.1539),
    'Cheshire, CT': (-72.9106, 41.5084)
}

with open("config.json") as f:
    ORS_API_KEY = json.load(f)["openrouteservice_api_key"]

import openrouteservice
client = openrouteservice.Client(key=ORS_API_KEY)

coords = [cities_coords[city] for city in ['Fresno, CA', 'Tulsa, OK', 'Cheshire, CT']]
route = client.directions(coords, profile='driving-car', format='geojson')
geom = route['features'][0]['geometry']
folium.GeoJson(geom, name="Fresno to Cheshire via Tulsa", style_function=lambda x: {'color': 'blue', 'weight': 3}).add_to(m)

# Add each station to the map with color based on fuel price
min_price = df['Best Discounted Price'].min()
max_price = df['Best Discounted Price'].max()

for _, row in df.iterrows():
    norm_price = (row['Best Discounted Price'] - min_price) / (max_price - min_price)
    rgba = plt.cm.YlOrRd(norm_price)
    color = mcolors.rgb2hex(rgba)

    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        tooltip=(f"Store No: {row['Store No']}<br>"
                 f"City: {row['City']}, {row['State']}<br>"
                 f"Price: ${row['Best Discounted Price']:.2f}<br>"
                 f"Distance from Fresno: {row['Distance from Fresno (miles)']:.1f} mi")
    ).add_to(m)

# Save to HTML
m.save("fresno_to_cheshire_stations_map.html")
print("✅ Map saved as fresno_to_cheshire_stations_map.html")
