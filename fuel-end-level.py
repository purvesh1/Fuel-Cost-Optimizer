# Streamlit version of route planner with dynamic input UI
import streamlit as st
import folium
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from geopy.distance import geodesic
from shapely.geometry import Point, LineString
import openrouteservice
from streamlit_folium import st_folium
import json
import yaml

# Load station data
stations_df = pd.read_excel("prcqte.xlsx", sheet_name="LovesPrices")
stations_gdf = gpd.GeoDataFrame(
    stations_df,
    geometry=gpd.points_from_xy(stations_df.Longitude, stations_df.Latitude),
    crs="EPSG:4326"
)

# Coordinates for cities
cities_coords = {
    'Fresno, CA': (-119.7871, 36.7378),
    'Cheshire, CT': (-72.9106, 41.5084),
    'Maryland': (-76.6413, 39.0458),
    'Chicago, IL': (-87.6298, 41.8781),
    'Sacramento, CA': (-121.4944, 38.5816),
    'Los Angeles, CA': (-118.2437, 34.0522),
    'Atlanta, GA': (-84.3880, 33.7490),
    'Tulsa, OK': (-95.9928, 36.1539)
}

routes = {
    'Fresno to Cheshire via Tulsa': ['Fresno, CA', 'Tulsa, OK', 'Cheshire, CT'],
    'Cheshire to Fresno via Tulsa': ['Cheshire, CT', 'Tulsa, OK', 'Fresno, CA'],
    'Fresno to Maryland': ['Fresno, CA', 'Maryland'],
    'Maryland to Fresno': ['Maryland', 'Fresno, CA'],
    'Fresno to Chicago': ['Fresno, CA', 'Chicago, IL'],
    'Chicago to Fresno': ['Chicago, IL', 'Fresno, CA'],
    'Sacramento to Chicago': ['Sacramento, CA', 'Chicago, IL'],
    'Chicago to Sacramento': ['Chicago, IL', 'Sacramento, CA'],
    'LA to Chicago': ['Los Angeles, CA', 'Chicago, IL'],
    'Chicago to LA': ['Chicago, IL', 'Los Angeles, CA'],
    'Fresno to Atlanta': ['Fresno, CA', 'Atlanta, GA'],
    'Atlanta to Fresno': ['Atlanta, GA', 'Fresno, CA']
}

# Sidebar inputs
st.sidebar.title("Fuel Route Optimizer")
selected_city = st.sidebar.selectbox("Select Starting City", list(cities_coords.keys()))
fuel_avg = st.sidebar.slider("Fuel Average (miles/gal)", 5.0, 18.0, 7.3)
tank_capacity = st.sidebar.slider("Tank Capacity (gal)", 50.0, 300.0, 240.0)
minimum_fuel = st.sidebar.slider("Minimum Reserve Fuel (gal)", 0.0, 220.0, 50.0)

# OpenRouteService client

# Load API key from YAML file
with open('keys.yaml', 'r') as file:
    keys = yaml.safe_load(file)
ORS_API_KEY = keys['ors']['api_key']
ORS_API_KEY = "5b3ce3597851110001cf624843979a6265694fbfbb601f1e94a86856"  # Replace with your API key
client = openrouteservice.Client(key=ORS_API_KEY)

# Price range for color scale
price_min = stations_gdf['Best Discounted Price'].min()
price_max = stations_gdf['Best Discounted Price'].max()

# Folium map
m = folium.Map(location=[39.8283, -98.5795], zoom_start=5)

# Filter routes starting from selected city
selected_routes = {k: v for k, v in routes.items() if selected_city == v[0]}

for name, cities in selected_routes.items():
    coords = [cities_coords[city] for city in cities]
    route = client.directions(coordinates=coords, profile='driving-car', format='geojson')
    geom = route['features'][0]['geometry']

    folium.GeoJson(geom, name=name, style_function=lambda x: {'color': 'blue', 'weight': 3}).add_to(m)

    line = LineString(geom['coordinates'])
    buffer = line.buffer(500 / 111139)  # ~50km buffer
    stations_gdf['on_this_route'] = stations_gdf.geometry.within(buffer)
    stations = stations_gdf[stations_gdf['on_this_route']].copy()

    stations['distance'] = stations.geometry.apply(
        lambda x: geodesic((cities_coords[selected_city][1], cities_coords[selected_city][0]), (x.y, x.x)).miles)

    max_range = (tank_capacity - minimum_fuel) * fuel_avg
    reachable = stations[stations['distance'] <= max_range]

    if not reachable.empty:
        cheapest = reachable.loc[reachable['Best Discounted Price'].idxmin()]
        fuel_required = (cheapest['distance'] / fuel_avg) + minimum_fuel

        st.subheader(f"{name}")
        st.markdown(f"**Cheapest Stop:** {cheapest['City']}, {cheapest['State']} at ${cheapest['Best Discounted Price']:.2f}")
        st.markdown(f"**Distance:** {cheapest['distance']:.2f} miles")
        st.markdown(f"**Required Starting Fuel:** {fuel_required:.2f} gallons")

    for _, row in stations.iterrows():
        norm = (row['Best Discounted Price'] - price_min) / (price_max - price_min)
        color = mcolors.rgb2hex(plt.cm.YlOrRd(norm))
        folium.CircleMarker(
            location=(row.geometry.y, row.geometry.x),
            radius=5,
            popup=f"{row['City']}, {row['State']}\n${row['Best Discounted Price']:.2f}",
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7
        ).add_to(m)


# Display map
st_data = st_folium(m, width=900, height=600)
