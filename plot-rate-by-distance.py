import streamlit as st
import folium
from streamlit_folium import st_folium
import json
import openrouteservice
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

# Load API key from config.json
with open("config.json", "r") as file:
    config = json.load(file)

ORS_API_KEY = config['openrouteservice_api_key']
client = openrouteservice.Client(key=ORS_API_KEY)

# City coordinates
cities_coords = {
    'Fresno, CA': (-119.7871, 36.7378),
    'Cheshire, CT': (-72.9106, 41.5084),
    'Maryland': (-76.6413, 39.0458),
    'Chicago, IL': (-87.6298, 41.8781),
    'Sacramento, CA': (-121.4944, 38.5816),
    'Los Angeles, CA': (-118.2437, 34.0522),
    'Atlanta, GA': (-84.3880, 33.7490)
}

# Define routes and reverse routes
routes = {
    'Fresno to Cheshire': ['Fresno, CA', 'Cheshire, CT'],
    'Cheshire to Fresno': ['Cheshire, CT', 'Fresno, CA'],
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

# Select route
st.title("Interactive Route Distance Picker")
route_name = st.selectbox("Choose a route:", list(routes.keys()))

selected_route = routes[route_name]
coords = [cities_coords[city] for city in selected_route]
route = client.directions(coords, profile='driving-car', format='geojson')
route_line = LineString(route['features'][0]['geometry']['coordinates'])

# Folium map setup
m = folium.Map(location=[39.8283, -98.5795], zoom_start=5)
folium.GeoJson(route['features'][0]['geometry'], name="Route").add_to(m)

# Allow user to click two points
clicks = st_folium(m, width=900, height=600, returned_objects=["last_clicked"], key="map")

if 'clicked_points' not in st.session_state:
    st.session_state.clicked_points = []

if clicks and clicks["last_clicked"]:
    latlon = clicks["last_clicked"]
    st.session_state.clicked_points.append((latlon["lng"], latlon["lat"]))

# Process after two clicks
if len(st.session_state.clicked_points) == 2:
    pt1 = Point(st.session_state.clicked_points[0])
    pt2 = Point(st.session_state.clicked_points[1])

    # Snap both points to route
    snapped1 = nearest_points(route_line, pt1)[0]
    snapped2 = nearest_points(route_line, pt2)[0]

    # Measure distance along route
    coords = list(route_line.coords)
    dist = 0
    found_start = False

    for i in range(len(coords) - 1):
        seg_start = Point(coords[i])
        seg_end = Point(coords[i + 1])
        segment = LineString([seg_start, seg_end])

        if not found_start and segment.distance(snapped1) < 1e-4:
            dist += segment.length
            found_start = True
        elif found_start:
            dist += segment.length
            if segment.distance(snapped2) < 1e-4:
                break

    dist_miles = dist * 69.172  # degrees to miles approximation
    st.success(f"Distance along {route_name}: {dist_miles:.2f} miles")

    # Reset points
    st.session_state.clicked_points = []
