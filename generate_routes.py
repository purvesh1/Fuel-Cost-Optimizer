# Required libraries
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import openrouteservice
from tqdm import tqdm
import json
import os

from route_data import cities_coords, routes

# Load config for API key
with open("config.json") as f:
    ORS_API_KEY = json.load(f)["openrouteservice_api_key"]

client = openrouteservice.Client(key=ORS_API_KEY)

# Load station data
stations_df = pd.read_excel("prcqte.xlsx", sheet_name="LovesPrices")
stations_gdf = gpd.GeoDataFrame(
    stations_df,
    geometry=gpd.points_from_xy(stations_df.Longitude, stations_df.Latitude),
    crs="EPSG:4326"
)

# Process each route
for route_name, route_info in routes.items():
    print(f"Processing: {route_name}")
    route_coords = [cities_coords[city] for city in route_info['cities']]
    start_coords = route_coords[0]
    end_coords = route_coords[-1]
    output_file = route_info["default_file"]

    try:
        # Get full route
        route = client.directions(route_coords, profile='driving-car', format='geojson')
        geom = route['features'][0]['geometry']
        line = LineString(geom['coordinates'])
        buffer = line.buffer(300 / 111139)

        # Filter stations along buffer
        stations_gdf['on_route'] = stations_gdf.geometry.within(buffer)
        route_stations = stations_gdf[stations_gdf['on_route']].copy()

        # Compute distance from start to each station
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
                    'Distance from Start (miles)': round(distance_miles, 2)
                })
            except Exception as e:
                print(f"Error with store {row['Loves Store No.']}: {e}")
                continue

        # Add endpoint
        end_distance_m = sum(seg['distance'] for seg in route['features'][0]['properties']['segments'])
        end_distance_miles = end_distance_m / 1609.34

        df = pd.DataFrame(records)
        df = pd.concat([df, pd.DataFrame([{
            'Store No': -1,
            'City': route_info['cities'][-1].split(',')[0],
            'State': route_info['cities'][-1].split(',')[1].strip(),
            'Latitude': end_coords[1],
            'Longitude': end_coords[0],
            'Best Discounted Price': None,
            'Distance from Start (miles)': round(end_distance_miles, 2)
        }])], ignore_index=True)

        # Save file
        df = df.sort_values(by='Distance from Start (miles)').reset_index(drop=True)
        if os.path.exists(output_file):
            os.remove(output_file)
        df.to_excel(output_file, index=False)
        print(f"✅ Saved: {output_file}")

    except openrouteservice.exceptions.ApiError as e:
        print(f"❌ Failed for {route_name}: {e}")
