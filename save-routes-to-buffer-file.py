# Required libraries
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import openrouteservice
from tqdm import tqdm
import json
import os

# Load city coordinates
cities_coords = {
    'Fresno, CA': (-119.7871, 36.7378),
    'Tulsa, OK': (-95.9928, 36.1539),
    'Cheshire, CT': (-72.9106, 41.5084)
}

# Load config for API key
with open("config.json") as f:
    ORS_API_KEY = json.load(f)["openrouteservice_api_key"]

# OpenRouteService client setup
client = openrouteservice.Client(key=ORS_API_KEY)

# Load station data
stations_df = pd.read_excel("prcqte.xlsx", sheet_name="LovesPrices")
stations_gdf = gpd.GeoDataFrame(
    stations_df,
    geometry=gpd.points_from_xy(stations_df.Longitude, stations_df.Latitude),
    crs="EPSG:4326"
)

# Route: Fresno → Tulsa → Cheshire
route_coords = [cities_coords[city] for city in ['Fresno, CA', 'Tulsa, OK', 'Cheshire, CT']]
route = client.directions(route_coords, profile='driving-car', format='geojson')
geom = route['features'][0]['geometry']
line = LineString(geom['coordinates'])
buffer = line.buffer(300 / 111139)

# Filter stations along the route
stations_gdf['on_route'] = stations_gdf.geometry.within(buffer)
route_stations = stations_gdf[stations_gdf['on_route']].copy()

# Compute driving distance from Fresno to each station
start_coords = cities_coords['Fresno, CA']
records = []

for _, row in tqdm(route_stations.iterrows(), total=route_stations.shape[0]):
    try:
        station_coords = (row['Longitude'], row['Latitude'])
        route_to_station = client.directions(
            coordinates=[start_coords, station_coords],
            profile='driving-car',
            format='geojson'
        )
        distance_m = route_to_station['features'][0]['properties']['segments'][0]['distance']
        distance_miles = distance_m / 1609.34

        records.append({
            'Store No': row['Loves Store No.'],
            'City': row['City'],
            'State': row['State'],
            'Latitude': row['Latitude'],
            'Longitude': row['Longitude'],
            'Best Discounted Price': row['Best Discounted Price'],
            'Distance from Fresno (miles)': round(distance_miles, 2)
        })
    except Exception as e:
        print(f"Error with store {row['Loves Store No.']}: {e}")
        continue

# Create DataFrame
df = pd.DataFrame(records)

# Add destination as endpoint using segment logic
end_city = 'Cheshire, CT'
dest_coords = cities_coords[end_city]

end_distance_m = sum(segment['distance'] for segment in route['features'][0]['properties']['segments'])
end_distance_miles = end_distance_m / 1609.34


df = pd.concat([df, pd.DataFrame([{
    'Store No': -1,
    'City': 'Cheshire',
    'State': 'CT',
    'Latitude': dest_coords[1],
    'Longitude': dest_coords[0],
    'Best Discounted Price': None,
    'Distance from Fresno (miles)': round(end_distance_miles, 2)
}])], ignore_index=True)

# Sort and save
df = df.sort_values(by='Distance from Fresno (miles)').reset_index(drop=True)
excel_path = "fresno_to_cheshire_stations.xlsx"
if os.path.exists(excel_path):
    os.remove(excel_path)
df.to_excel(excel_path, index=False)
print("✅ Sorted Excel file saved: fresno_to_cheshire_stations.xlsx")
