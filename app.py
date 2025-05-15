# app.py
import folium
import matplotlib.colors
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import pulp
import math
import altair as alt
from route_data import routes, cities_coords # Assuming route_data.py is in the same directory
from streamlit_folium import st_folium

# --- START OF ADAPTED CORE LOGIC (from your provided script) ---
# --- (Normally this would be in fuel_optimizer_core.py and imported) ---

# --- 1. Data Structures (Using Gallons and Miles) ---
class Truck:
    def __init__(self, name, tank_capacity_gal, avg_consumption_gal_per_mile):
        self.name = name
        self.tank_capacity_gal = tank_capacity_gal
        self.avg_consumption_gal_per_mile = avg_consumption_gal_per_mile

class FuelStation: # Represents actual refueling stations
    def __init__(self, id, name, location_miles, price_per_gallon, latitude, longitude, stoppage_cost=0.0):
        self.id = str(id) 
        self.name = name
        self.location_miles = location_miles
        self.price_per_gallon = price_per_gallon
        self.latitude = latitude
        self.longitude = longitude
        self.stoppage_cost = stoppage_cost # Cost incurred for stopping at the station (if any)

    def __repr__(self):
        return (f"FuelStation(id='{self.id}', name='{self.name}', "
                f"location_miles={self.location_miles:.2f}, price_per_gallon={self.price_per_gallon:.3f}, "
                f"latitude={self.latitude:.3f}, longitude={self.longitude:.3f}),  stoppage_cost={self.stoppage_cost:.2f})")

class Route:
    def __init__(self, name, start_location_miles, end_location_miles, station_ids_in_order):
        self.name = name
        self.start_location_miles = start_location_miles
        self.end_location_miles = end_location_miles
        self.station_ids_in_order = [str(sid) for sid in station_ids_in_order]

# --- Helper function to load stations and identify endpoint from Excel ---
@st.cache_data
def load_stations_and_endpoint_from_excel(file_path, stoppage_cost, sheet_name=0, endpoint_store_no="-1"):
    log_messages = []
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        log_messages.append(f"Successfully read Excel file. Columns found: {df.columns.tolist()}")
    except FileNotFoundError:
        log_messages.append(f"Error: Excel file not found at {file_path}")
        st.error(f"Error: Excel file not found at {file_path}. Please upload a valid file.")
        return [], None, log_messages
    except Exception as e:
        log_messages.append(f"Error reading Excel file: {e}")
        return [], None, log_messages

    fuel_stations = []
    endpoint_details = None
    
    id_col = 'Store No'
    name_col_city = 'City'
    location_col_miles_excel = 'Distance from Start (miles)'
    price_col_excel = 'Best Discounted Price'
    latitude_col = 'Latitude'
    longitude_col = 'Longitude'

    required_cols = [id_col, name_col_city, location_col_miles_excel]
    if price_col_excel not in df.columns:
        log_messages.append(f"Warning: Price column '{price_col_excel}' not found. Stations will not have price data.")

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        log_messages.append(f"Error: Required column(s) {missing_cols} not found in Excel sheet.")
        log_messages.append(f"Please ensure file contains these columns with exact names.")
        return [], None, log_messages

    for index, row in df.iterrows():
        try:
            station_id_str = str(row[id_col]).strip()
            city_name = str(row[name_col_city]).strip()
            
            current_row_location_miles = pd.to_numeric(row[location_col_miles_excel], errors='coerce')
            if pd.isna(current_row_location_miles) or current_row_location_miles < 0:
                log_messages.append(f"Warning: Skipping row {index+2} (ID: {station_id_str}) due to invalid or negative distance: '{row[location_col_miles_excel]}'")
                continue

            if station_id_str == endpoint_store_no:
                if endpoint_details is not None:
                    log_messages.append(f"Warning: Multiple endpoint entries found with Store No '{endpoint_store_no}'. Using the first one encountered: '{endpoint_details['name']}'.")
                else:
                    endpoint_details = {'name': city_name if city_name else "Endpoint", 'location_miles': current_row_location_miles}
                    log_messages.append(f"Identified endpoint: Name='{endpoint_details['name']}', Location={endpoint_details['location_miles']:.2f} miles (from row {index+2})")
                continue 

            station_name = f"{city_name} (ID: {station_id_str})"
            price_per_gallon = 0.0 
            if price_col_excel in df.columns:
                price_val = pd.to_numeric(row[price_col_excel], errors='coerce')
                if pd.isna(price_val) or price_val <= 0:
                    log_messages.append(f"Warning: Station ID {station_id_str} (row {index+2}) has invalid or non-positive price: '{row[price_col_excel]}'. Assuming $0.0/Gal.")
                else:
                    price_per_gallon = price_val
            fuel_stations.append(FuelStation(
                id=station_id_str,
                name=station_name,
                location_miles=current_row_location_miles,
                price_per_gallon=price_per_gallon,
                latitude=row[latitude_col] if latitude_col in df.columns else None,
                longitude=row[longitude_col] if longitude_col in df.columns else None,
                stoppage_cost=stoppage_cost
            ))
        except KeyError as e:
            log_messages.append(f"Warning: Missing expected column for row {index+2}. Error: {e}. Skipping this row.")
            continue
        except Exception as e: 
            log_messages.append(f"Warning: Error processing row {index+2} (ID: {station_id_str}). Error: {e}. Skipping this row.")
            continue
            
    fuel_stations.sort(key=lambda s: s.location_miles)
    return fuel_stations, endpoint_details, log_messages

def get_station_by_id(station_id, all_stations):
    target_station_id = str(station_id)
    for station in all_stations:
        if station.id == target_station_id:
            return station
    return None

def optimize_fuel_stops(
    truck, # Assuming Truck object with attributes like tank_capacity_gal, avg_consumption_gal_per_mile
    route, # Assuming Route object with attributes like name, start_location_miles, end_location_miles, station_ids_in_order
    all_stations_data: list, # List of Station objects, assuming each has id, name, location_miles, price_per_gallon, latitude, longitude, and stoppage_cost
    start_fuel_gal: float,
    desired_end_fuel_gal: float,
    min_buffer_fuel_gal: float,
    max_refueling_stops: int, # Keep parameter in function signature, but it will be ignored
    log_area=None): # Pass streamlit element for logging

    start_city = routes[route.name]['cities'][0] if route.name in routes else "Unknown Start"
    end_city = routes[route.name]['cities'][-1] if route.name in routes else "Unknown End"

    op_log = [] # Optimization specific logs
    def _log(msg):
        op_log.append(msg)
        if log_area:
            log_area.info(msg) # Or .text, .write
        else:
            print(msg)

    # --- Prepare Points of Interest (POIs) ---
    pois = []
    # Add the start point
    # Note: Using start_city coords for start point lat/lon - might need adjustment if route start is not exactly the city center
    start_coords = cities_coords.get(start_city, (0, 0)) # Use .get for safety
    pois.append({
        'id': 'start',
        'name': f'Start Route ({start_city})',
        'location_miles': route.start_location_miles,
        'price_per_gallon': 0, # Price not relevant for start
        'stoppage_cost': 0, # Stoppage cost not relevant for start
        'latitude': float(start_coords[1]), # Assuming cities_coords stores (lon, lat)
        'longitude': float(start_coords[0]),
        'is_station': False
    })

    # Add relevant stations along the route
    route_specific_stations_objects = []
    if all_stations_data:
        for station_id_on_route in route.station_ids_in_order:
            station_obj = get_station_by_id(station_id_on_route, all_stations_data)
            if station_obj:
                 # Only consider stations strictly between the start and end locations for refueling stops
                if route.start_location_miles < station_obj.location_miles < route.end_location_miles:
                    route_specific_stations_objects.append(station_obj)
                elif station_obj.location_miles == route.end_location_miles:
                     _log(f"Info: Station {station_obj.name} (ID: {station_obj.id}) is at the exact destination location ({route.end_location_miles:.2f} miles) and will not be considered a refueling stop on the way.")
                elif station_obj.location_miles <= route.start_location_miles:
                     _log(f"Info: Station {station_obj.name} (ID: {station_obj.id}) at {station_obj.location_miles:.2f} miles is before or at the route's start ({route.start_location_miles:.2f} miles) and will be excluded.")
                elif station_obj.location_miles >= route.end_location_miles :
                    _log(f"Info: Station {station_obj.name} (ID: {station_obj.id}) at {station_obj.location_miles:.2f} miles is beyond or at the route's end ({route.end_location_miles:.2f} miles) and will be excluded.")
            else:
                _log(f"Warning: Station ID '{station_id_on_route}' defined in route '{route.name}' not found in the provided list of all_stations_data.")
    else:
        _log("Info: No fuel stations provided to select from for the route.")

    route_specific_stations_objects.sort(key=lambda s: s.location_miles)
    for station in route_specific_stations_objects:
        # Ensure the Station object has a stoppage_cost attribute
        stoppage_cost = station.stoppage_cost # Default to 0 if attribute not found
        pois.append({
            'id': station.id,
            'name': station.name,
            'location_miles': station.location_miles,
            'price_per_gallon': station.price_per_gallon,
            'stoppage_cost': stoppage_cost, # Include stoppage cost
            'latitude': float(station.latitude),
            'longitude': float(station.longitude),
            'is_station': True
        })

    # Add the end point
    # Using end_city coords for end point lat/lon
    end_coords = cities_coords.get(end_city, (0, 0)) # Use .get for safety
    pois.append({
        'id': 'end',
        'name': f'End Route ({end_city})',
        'location_miles': route.end_location_miles,
        'price_per_gallon': 0, # Price not relevant for end
        'stoppage_cost': 0, # Stoppage cost not relevant for end
        'latitude': float(end_coords[1]), # Assuming cities_coords stores (lon, lat)
        'longitude': float(end_coords[0]),
        'is_station': False
    })

    # Ensure unique POIs, prioritizing stations or start/end at the same location
    unique_pois_dict = {}
    for p in sorted(pois, key=lambda x: (x['location_miles'], not x['is_station'])):
        loc = p['location_miles']
        if loc not in unique_pois_dict:
            unique_pois_dict[loc] = p
        # If a station is at the exact same location as a non-station (like start/end),
        # prioritize the station if it's not the start/end itself.
        # Or if a start/end is at the same location as a non-station (which shouldn't happen if logic is right)
        elif p['is_station'] and not unique_pois_dict[loc]['is_station']:
             # If the existing POI at this location is not a station, replace it with the station
             unique_pois_dict[loc] = p
        elif (p['id'] == 'start' or p['id'] == 'end') and not unique_pois_dict[loc]['is_station']:
             # If the existing POI is not a station and the current one is start/end, keep the start/end
             unique_pois_dict[loc] = p
        # If both are stations or both are non-stations at the same location, the first one encountered (sorted) is kept.

    pois = sorted(list(unique_pois_dict.values()), key=lambda p_item: p_item['location_miles'])

    num_pois = len(pois)
    if num_pois < 2:
        _log("Error: Route must have at least a start and end point after processing POIs.")
        return None, op_log, None, None

    # Identify indices of actual fuel stations in the final sorted POI list
    actual_station_poi_indices = [i for i, p_item in enumerate(pois) if p_item['is_station']]
    _log(f"Number of Points of Interest (POIs) for optimization: {num_pois}")
    _log(f"Number of actual fuel stations among POIs: {len(actual_station_poi_indices)}")

    # --- Set up the PuLP problem ---
    prob = pulp.LpProblem(f"FuelOptimization_{truck.name}_{route.name}", pulp.LpMinimize)

    # Decision variables
    fuel_purchased_gal = pulp.LpVariable.dicts("FuelPurchasedGal", range(num_pois), lowBound=0, cat='Continuous')
    fuel_at_arrival_gal = pulp.LpVariable.dicts("FuelAtArrivalGal", range(num_pois), lowBound=min_buffer_fuel_gal, upBound=truck.tank_capacity_gal, cat='Continuous')
    fuel_after_purchase_gal = pulp.LpVariable.dicts("FuelAfterPurchaseGal", range(num_pois), lowBound=min_buffer_fuel_gal, upBound=truck.tank_capacity_gal, cat='Continuous')

    # Binary variable for stop decision (only for actual stations)
    stop_decision = {}
    if actual_station_poi_indices:
        stop_decision = pulp.LpVariable.dicts("StopDecision", actual_station_poi_indices, cat='Binary')
    else:
        _log("Info: No actual fuel stations identified on the route for optimization variables.")


    # --- Objective Function: Minimize Total Cost (Fuel Cost + Stoppage Cost) ---
    total_cost = pulp.LpAffineExpression()
    for i in range(num_pois):
        if pois[i]['is_station']:
            # Add fuel purchase cost
            if i not in actual_station_poi_indices:
                 _log(f"CRITICAL LOGIC ERROR: POI {i} ('{pois[i]['name']}') is_station=True but not in actual_station_poi_indices. Skipping cost calculation for this POI.")
                 continue
            total_cost += fuel_purchased_gal[i] * pois[i]['price_per_gallon']
            # Add stoppage cost if a stop is made
            if i in stop_decision: # Ensure stop_decision variable exists for this index
                 total_cost += pois[i]['stoppage_cost'] * stop_decision[i]
            else:
                 _log(f"Warning: POI {i} ('{pois[i]['name']}') is a station but no stop_decision variable found. Stoppage cost will not be included for this station.")

    prob += total_cost, "TotalFuelAndStoppageCost"


    # --- Constraints ---

    # Initial fuel at the start
    if not (pois[0]['id'] == 'start' and pois[0]['location_miles'] == route.start_location_miles):
        _log(f"Error: First POI is not the route start. POI[0]: {pois[0]}. Route Start: {route.start_location_miles}")
        return None, op_log, None, None
    prob += fuel_at_arrival_gal[0] == start_fuel_gal, "InitialFuelConstraint"
    prob += fuel_after_purchase_gal[0] == fuel_at_arrival_gal[0], "NoPurchaseAtStartConstraint" # Cannot purchase fuel at the start point
    prob += fuel_purchased_gal[0] == 0, f"ZeroPurchaseAtStartPOI"


    for i in range(num_pois):
        # Fuel balance after purchase (only for stations)
        if pois[i]['is_station']:
             if i not in actual_station_poi_indices or i not in stop_decision:
                 _log(f"Critical Error during constraint setup: POI {i} ({pois[i]['name']}) is station but has inconsistent setup for stop_decision. Skipping constraints for this POI.")
                 continue
             prob += fuel_after_purchase_gal[i] == fuel_at_arrival_gal[i] + fuel_purchased_gal[i], f"FuelBalanceAfterPurchase_{pois[i]['id']}"

             # Link purchase amount to stop decision for stations
             prob += fuel_purchased_gal[i] <= truck.tank_capacity_gal * stop_decision[i], f"LinkPurchaseToStopDecisionUpper_{pois[i]['id']}"
             # If stop_decision is 1, must purchase at least a minimal amount (e.g., 0.01 gal) to avoid numerical issues
             prob += fuel_purchased_gal[i] >= 0.01 * stop_decision[i], f"LinkPurchaseToStopDecisionLower_{pois[i]['id']}"
             # Cannot purchase more than tank capacity minus fuel on arrival
             prob += fuel_purchased_gal[i] <= truck.tank_capacity_gal - fuel_at_arrival_gal[i], f"PurchaseCapacityLimit_{pois[i]['id']}"
        else:
            # No purchase at non-station points (start/end)
            prob += fuel_after_purchase_gal[i] == fuel_at_arrival_gal[i], f"NoPurchaseNonStation_{pois[i]['id']}"
            prob += fuel_purchased_gal[i] == 0, f"ZeroPurchaseNonStation_{pois[i]['id']}"


        # Fuel flow between consecutive points
        if i < num_pois - 1:
            dist_to_next_miles = pois[i+1]['location_miles'] - pois[i]['location_miles']
            if dist_to_next_miles < 0:
                _log(f"Error: Negative distance from POI {i} ('{pois[i]['name']}') to POI {i+1} ('{pois[i+1]['name']}'). Dist: {dist_to_next_miles:.2f} miles. POIs not sorted correctly.")
                return None, op_log, None, None

            fuel_needed_for_segment_gal = dist_to_next_miles * truck.avg_consumption_gal_per_mile
            prob += fuel_at_arrival_gal[i+1] == fuel_after_purchase_gal[i] - fuel_needed_for_segment_gal, f"FuelFlow_{pois[i]['id']}_to_{pois[i+1]['id']}"

            # Must have enough fuel to reach the next point with buffer
            prob += fuel_after_purchase_gal[i] >= fuel_needed_for_segment_gal + min_buffer_fuel_gal, f"SufficientFuelForSegment_{pois[i]['id']}_to_{pois[i+1]['id']}"


    # Desired fuel at the end point
    if not (pois[num_pois-1]['id'] == 'end' and pois[num_pois-1]['location_miles'] == route.end_location_miles):
        _log(f"Error: Last POI is not the route end. POI[{num_pois-1}]: {pois[num_pois-1]}. Route End: {route.end_location_miles}")
        return None, op_log, None, None
    prob += fuel_at_arrival_gal[num_pois-1] >= desired_end_fuel_gal, "DesiredEndFuelConstraint"

    # Maximum number of refueling stops
    # REMOVED as requested: prob += pulp.lpSum(stop_decision[idx] for idx in actual_station_poi_indices) <= max_refueling_stops, "MaxStopsConstraint"
    # Added a log message indicating the constraint is removed if max_refueling_stops is provided
    if actual_station_poi_indices and max_refueling_stops > 0:
         _log(f"Info: Maximum refueling stops constraint ({max_refueling_stops}) is ignored as requested.")
    elif not actual_station_poi_indices and max_refueling_stops > 0:
         _log(f"Info: Max stops constraint ({max_refueling_stops}) is set, but no fuel stations are part of the optimization POIs. Constraint is trivially satisfied.")


    # --- Solve the problem ---
    solver_to_use = pulp.PULP_CBC_CMD(msg=0) # Use CBC solver, suppress output
    try:
        status = prob.solve(solver_to_use)
    except pulp.apis.core.PulpSolverError as e:
        _log(f"PulpSolverError occurred: {e}. Ensure solver (CBC) is installed and in PATH.")
        return None, op_log, None, None
    except Exception as e:
         _log(f"An unexpected error occurred during solving: {e}")
         return None, op_log, None, None


    # --- Process Results ---
    if pulp.LpStatus[status] == "Optimal":
        _log("\nOptimal solution found!")

        # Calculate the actual number of stops made
        stops_made_val = 0
        if actual_station_poi_indices and stop_decision:
             # Ensure we only sum values for indices that are actually in stop_decision
             stops_made_val = sum(pulp.value(stop_decision[idx]) for idx in actual_station_poi_indices if idx in stop_decision and pulp.value(stop_decision[idx]) is not None)
             # Round the stop count as it should be an integer from the binary variable
             stops_made_val = round(stops_made_val)

        print(f"################### fuel_cost: {pulp.value(total_cost) - stops_made_val*stoppage_cost}") # Keep max allowed input in log for context

        solution = {
            'total_cost': pulp.value(prob.objective), # This now includes fuel cost + stoppage costs
            'fuel_cost': pulp.value(total_cost) - stops_made_val*stoppage_cost, # Total cost from the objective function
            'truck': truck.name,
            'route': route.name,
            'stops_made_count': stops_made_val,
            'total_fuel_filled': 0,
            'stops': [] # List to detail the actual refueling stops
        }

        _log(f"Total Trip Cost (Fuel + Stops): ${solution['total_cost']:.2f}")
        _log(f"Number of refueling stops made: {solution['stops_made_count']:.0f} (Max allowed input: {max_refueling_stops})") # Keep max allowed input in log for context

        # Prepare detailed plan for display/table
        detailed_plan_for_table = []
        # Header row for the table
        header_plan = {'Location':'Location', 'Distance (Miles)':'Distance (Miles)', 'Arrival Fuel (Gal)':'Arrival Fuel (Gal)', 'Purchased Fuel (Gal)':'Purchased Fuel (Gal)', 'Depart Fuel (Gal)':'Depart Fuel (Gal)', 'Fuel Cost ($)':'Fuel Cost ($)', 'Stoppage Cost ($)':'Stoppage Cost ($)', 'Total Cost at POI ($)':'Total Cost at POI ($)', 'Price ($/Gal)':'Price ($/Gal)'}
        detailed_plan_for_table.append(header_plan)


        for i in range(num_pois):
            poi_name = pois[i]['name']
            poi_location_miles = pois[i]['location_miles']

            # Get values from LP variables, handle None if optimization failed partially (though status is Optimal)
            arr_fuel_val = pulp.value(fuel_at_arrival_gal[i])
            arr_fuel = arr_fuel_val if arr_fuel_val is not None else 0.0

            pur_fuel = 0.0
            fuel_cost_at_stop = 0.0
            stoppage_cost_at_stop = 0.0
            total_cost_at_poi = 0.0
            price_at_stop_str = "-"
            is_stop_decision_one = False # Flag to check if a stop was decided at this station

            if pois[i]['is_station']:
                var_value = pulp.value(fuel_purchased_gal[i])
                if var_value is not None:
                    pur_fuel = var_value
                else:
                    _log(f"Warning: fuel_purchased_gal[{i}] value is None for station {poi_name}. Assuming 0.")

                price_at_stop = pois[i]['price_per_gallon']
                price_at_stop_str = f"{price_at_stop:.3f}"

                # Check the stop decision binary variable value
                if i in stop_decision:
                    stop_decision_val = pulp.value(stop_decision[i])
                    if stop_decision_val is not None and round(stop_decision_val) == 1: # Round to handle potential floating point results near 0 or 1
                        is_stop_decision_one = True

                # Calculate costs only if a stop was decided and fuel was purchased (or it's a stop with zero purchase, though less likely)
                # The LP constraints link purchase to stop_decision, so if stop_decision is 1, purchase should be > 0.001
                if is_stop_decision_one: # or pur_fuel > 0.001: # Consider a stop made if decision is 1 OR minimal fuel purchased
                     fuel_cost_at_stop = pur_fuel * price_at_stop
                     stoppage_cost_at_stop = pois[i]['stoppage_cost'] # Add the fixed stoppage cost
                     total_cost_at_poi = fuel_cost_at_stop + stoppage_cost_at_stop


            dep_fuel_val = pulp.value(fuel_after_purchase_gal[i])
            dep_fuel = dep_fuel_val if dep_fuel_val is not None else 0.0

            # Append data for the detailed plan table
            detailed_plan_for_table.append({
                'Location':poi_name,
                'Distance (Miles)':f"{poi_location_miles:.2f}",
                'Arrival Fuel (Gal)':f"{arr_fuel:.2f}",
                'Purchased Fuel (Gal)':f"{pur_fuel:.2f}",
                'Depart Fuel (Gal)':f"{dep_fuel:.2f}",
                'Fuel Cost ($)':f"{fuel_cost_at_stop:.2f}",
                'Stoppage Cost ($)':f"{stoppage_cost_at_stop:.2f}",
                'Total Cost at POI ($)':f"{total_cost_at_poi:.2f}",
                'Price ($/Gal)':price_at_stop_str
            })

            # Add details to the 'stops' list if a stop was made
            if is_stop_decision_one and pur_fuel > 0.001 : # Only include stops where fuel was actually purchased (more than minimal)
                 solution['stops'].append({
                    'station_id': pois[i]['id'],
                    'station_name': poi_name,
                    'location_miles': pois[i]['location_miles'],
                    'fuel_at_arrival_gal': round(arr_fuel,2),
                    'fuel_purchased_gal': round(pur_fuel,2),
                    'fuel_after_purchase_gal': round(dep_fuel,2),
                    'price_per_gallon': pois[i]['price_per_gallon'],
                    'stoppage_cost_at_stop': round(stoppage_cost_at_stop, 2), # Include stoppage cost in stop details
                    'cost_for_stop': round(total_cost_at_poi,2), # Total cost for this stop
                    'station_latitude': pois[i]['latitude'],
                    'station_longitude': pois[i]['longitude']
                 })
                 solution['total_fuel_filled'] += pur_fuel # Accumulate total fuel filled

        solution['full_plan_table_data'] = detailed_plan_for_table

        # Return the solution, logs, processed POIs, and LP variables
        pulp_vars_values = {
            'arrival': {i: pulp.value(fuel_at_arrival_gal[i]) for i in range(num_pois)},
            'after_purchase': {i: pulp.value(fuel_after_purchase_gal[i]) for i in range(num_pois)},
            'purchased': {i: pulp.value(fuel_purchased_gal[i]) for i in range(num_pois)},
            'stop_decision': {i: pulp.value(stop_decision[i]) for i in actual_station_poi_indices if i in stop_decision} # Store values for stop decisions
        }
        # Note: We store the *values* of the LP variables, not the PuLP variable objects themselves,
        # as PuLP objects are not designed to be pickled/stored in session state.

        return solution, op_log, pois, pulp_vars_values # Return variable values

    else:
        # Optimization failed
        _log(f"Optimization failed. Status: {pulp.LpStatus[status]}")
        # Optional: Write the LP problem to a file for debugging
        # debug_file = f"fuel_problem_debug_{truck.name}_{route.name.replace(' ','_')}.lp"
        # try: prob.writeLP(debug_file); _log(f"Problem written to {debug_file}.")
        # except Exception as e_lp: _log(f"Could not write LP file: {e_lp}")
        return None, op_log, None, None # Return None for solution and vars on failure



# --- Streamlit App Starts Here ---
st.set_page_config(layout="wide")
st.title("🚚 Truck Fuel Optimization Tool")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Input Parameters")

    #uploaded_file = ""
    
    selected_route_name = st.selectbox("Choose a Route:", list(routes.keys())) 
    default_excel_file = routes[selected_route_name]['default_file'] if selected_route_name in routes else None
    # st.info(f"Default Excel file for this route: {default_excel_file}")
   
    endpoint_store_no = -1 #st.text_input("Endpoint 'Store No' in Excel", "-1", help="The 'Store No' that identifies your destination in the Excel file.")

    st.subheader("Truck Parameters")
    truck_name = 'Frightliner' #st.text_input("Truck Name/ID", "FreightlinerImperial")
    truck_tank_cap = st.number_input("Tank Capacity (Gallons)", min_value=10.0, max_value=500.0, value=240.0, step=10.0)
    truck_mpg = st.number_input("Fuel Efficiency (Miles per Gallon - MPG)", min_value=1.0, max_value=20.0, value=7.0, step=0.1)
    gpm = 1.0 / truck_mpg if truck_mpg > 0 else 0.2 # gallons per mile

    st.subheader("Route Parameters")
    origin_miles = 0.0 # Assuming Fresno is always the start at 0 miles as per Excel column.
    start_fuel = st.number_input("Starting Fuel (Gallons)", min_value=0.0, max_value=truck_tank_cap, value=100.0, step=5.0)
    desired_end_fuel = st.number_input("Desired Fuel at Destination (Gallons)", min_value=0.0, max_value=truck_tank_cap, value=150.0, step=5.0)
    safety_buffer = st.number_input("Safety Buffer Fuel (Gallons)", min_value=0.0, max_value=truck_tank_cap/2, value=70.0, step=5.0)
    stoppage_cost = st.number_input("Stoppage cost", min_value=0, max_value=1000, value=10, step=1)

    run_button = st.button("Run Optimization")

# --- Main Area for Results ---
st.header("Optimization Results")
log_expander = st.expander("Show Logs")
log_placeholder = log_expander.empty() # For dynamic log updates if needed, or just display all at end

if run_button:
    all_logs = []
    log_placeholder.info("Starting process...")

    excel_to_load = default_excel_file
    if default_excel_file is None :
        all_logs.append(f"No file uploaded, using default: {default_excel_file}")
    
    all_fuel_stations, city_endpoint, load_logs = load_stations_and_endpoint_from_excel(excel_to_load, stoppage_cost)
    all_logs.extend(load_logs)

    if not city_endpoint:
        all_logs.append(f"CRITICAL: Endpoint (Store No = {endpoint_store_no}) not found. Cannot define route.")
        st.error(f"CRITICAL: Endpoint (Store No = {endpoint_store_no}) not found. Cannot define route. Check logs.")
        log_placeholder.text("\n".join(all_logs))
        st.stop() # Stop further execution for this run

    all_logs.append(f"Detected Endpoint: Name='{city_endpoint['name']}', Location={city_endpoint['location_miles']:.2f} miles.")
    all_logs.append(f"Total actual fuel stations loaded: {len(all_fuel_stations)}")
    if not all_fuel_stations:
         all_logs.append(f"Warning: No actual fuel stations were loaded. Check data and column names.")

    my_truck = Truck(name=truck_name, tank_capacity_gal=truck_tank_cap, avg_consumption_gal_per_mile=gpm)

    route_end_miles = city_endpoint['location_miles']
    station_ids_for_route = [s.id for s in all_fuel_stations if s.location_miles < route_end_miles and s.location_miles >= origin_miles]
    
    current_route = Route(
        name=selected_route_name,
        start_location_miles=origin_miles,
        end_location_miles=route_end_miles,
        station_ids_in_order=station_ids_for_route
    )
    all_logs.append(f"Defined Route: {current_route.name} from {current_route.start_location_miles:.2f} miles to {current_route.end_location_miles:.2f} miles (Destination: {city_endpoint['name']})")
    all_logs.append(f"Number of fuel stations selected for this route (before destination): {len(station_ids_for_route)}")
    if not station_ids_for_route and current_route.end_location_miles > 0 :
         all_logs.append("Warning: No fuel stations identified on the route before the destination.")


    all_logs.append(f"Truck: {my_truck.name}, Capacity={my_truck.tank_capacity_gal:.1f} Gal, Consumption={my_truck.avg_consumption_gal_per_mile:.4f} Gal/Mile ({1/my_truck.avg_consumption_gal_per_mile:.1f} MPG)")
    all_logs.append(f"Fuel Parameters: Start={start_fuel:.1f} Gal, Desired End at {city_endpoint['name']}={desired_end_fuel:.1f} Gal, Safety Buffer={safety_buffer:.1f} Gal")
    
    log_placeholder.text("\n".join(all_logs) + "\nRunning PuLP optimizer...")

    # --- Run Optimization ---
    solution_data, opt_logs, pois_used, pulp_vars = optimize_fuel_stops(
        truck=my_truck,
        route=current_route,
        all_stations_data=all_fuel_stations,
        start_fuel_gal=start_fuel,
        desired_end_fuel_gal=desired_end_fuel,
        min_buffer_fuel_gal=safety_buffer,
        max_refueling_stops=10, # Ignored in the function, but kept for clarity
    )
    all_logs.extend(opt_logs)

    if solution_data and pois_used and pulp_vars:
        st.session_state['solution_data'] = solution_data
        st.session_state['pois_used'] = pois_used
        st.session_state['pulp_vars'] = pulp_vars
        st.session_state['current_route.name'] = current_route.name
        st.session_state['current_route.start_location_miles'] = current_route.start_location_miles
        st.session_state['current_route.end_location_miles'] = current_route.end_location_miles
        st.session_state['current_route_flag'] = bool(current_route)
        st.session_state['city_endpoint'] = city_endpoint
        st.session_state['all_logs'] = all_logs
        st.session_state['truck_tank_cap'] = truck_tank_cap
        st.session_state['safety_buffer'] = safety_buffer
        st.session_state['desired_end_fuel'] = desired_end_fuel
        st.session_state['stoppage_cost'] = stoppage_cost
        st.session_state['start_fuel'] = start_fuel
        st.session_state['len_stops'] = len(solution_data['stops'])

    else:
        st.error("Optimization failed or no solution found. Check logs for details.")
        if current_route: # Basic info if route was defined
            fuel_needed_approx = (current_route.end_location_miles * my_truck.avg_consumption_gal_per_mile) 
            all_logs.append(f"Approx fuel needed for route distance: {fuel_needed_approx:.2f} Gallons")
            max_range_on_start_fuel_approx = (start_fuel - safety_buffer) / my_truck.avg_consumption_gal_per_mile if my_truck.avg_consumption_gal_per_mile > 0 else 0
            all_logs.append(f"Approx max range on start fuel (with buffer): {max_range_on_start_fuel_approx:.2f} Miles")

    log_placeholder.text("\n".join(all_logs)) # Display all collected logs at the end
    all_logs.append("\n--- Script finished for this run ---")
        

else:
    st.info("Adjust parameters in the sidebar and click 'Run Optimization'.")


if 'solution_data' in st.session_state and 'pois_used' in st.session_state:
    solution_data = st.session_state['solution_data']
    pois_used = st.session_state['pois_used']
    pulp_vars = st.session_state['pulp_vars']
    current_route_name = st.session_state['current_route.name']
    current_route_start_miles = st.session_state['current_route.start_location_miles']
    current_route_end_miles = st.session_state['current_route.end_location_miles']
    current_route_flag =st.session_state['current_route_flag']
    city_endpoint = st.session_state['city_endpoint']
    all_logs = st.session_state['all_logs']
    truck_tank_cap = st.session_state['truck_tank_cap']
    safety_buffer = st.session_state['safety_buffer']
    desired_end_fuel = st.session_state['desired_end_fuel']
    stoppage_cost = st.session_state['stoppage_cost']
    start_fuel = st.session_state['start_fuel']
    len_stops = st.session_state['len_stops']

    
    # Show success message
    st.success("Optimization Successful!")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Fuel Cost", f"${solution_data['fuel_cost']:.2f}")
    col2.metric("Total Gallons", f"{solution_data['total_fuel_filled']:.2f}")
    col3.metric("No. of Stops", f"{len_stops}")

    st.subheader("Fueling Plan Details")
    # Convert list of dicts (with header as first dict) to DataFrame for st.table
    if solution_data.get('full_plan_table_data'):
        plan_df_data = solution_data['full_plan_table_data']
        if len(plan_df_data) > 1:
            plan_df = pd.DataFrame(plan_df_data[1:], columns=plan_df_data[0].values())
            st.dataframe(plan_df)
        else:
            st.info("Full plan table data is empty or header only.")
    else:
        st.info("No detailed plan table data in solution.")


    # --- Charting ---
    st.subheader("Fuel Level Along Route")
    
    # Prepare data for Altair chart
    chart_data_points = []
    # Consumption segments and refuel events
    for i in range(len(pois_used)):
        dist_current = pois_used[i]['location_miles']
        fuel_arrival = pulp.value(pulp_vars['arrival'][i])
        fuel_departure = pulp.value(pulp_vars['after_purchase'][i])
        
        if i > 0: # Arrival at POI (from previous departure)
                chart_data_points.append({'Distance': dist_current, 'Fuel': fuel_arrival, 'Series': 'Actual Fuel', 'Event': 'Arrival', 'Label': pois_used[i]['name']})

        if pois_used[i]['is_station'] and pulp.value(pulp_vars['purchased'][i]) > 0.001:
            # Point before refuel
            chart_data_points.append({'Distance': dist_current, 'Fuel': fuel_arrival, 'Series': 'Actual Fuel', 'Event': 'Refuel Start', 'Label': f"Buy {pulp.value(pulp_vars['purchased'][i]):.2f}G @ {pois_used[i]['name']}"})
            # Point after refuel
            chart_data_points.append({'Distance': dist_current, 'Fuel': fuel_departure, 'Series': 'Actual Fuel', 'Event': 'Refuel End', 'Label': pois_used[i]['name']})
        
        # Departure point for segment to next POI
        if i < len(pois_used) - 1:
            chart_data_points.append({'Distance': dist_current, 'Fuel': fuel_departure, 'Series': 'Actual Fuel', 'Event': 'Departure', 'Label': pois_used[i]['name']})
        elif i == len(pois_used) -1 : # Last point (destination arrival)
                chart_data_points.append({'Distance': dist_current, 'Fuel': fuel_arrival, 'Series': 'Actual Fuel', 'Event': 'Destination', 'Label': pois_used[i]['name']})


    # Add Start Point explicitly if not covered
    if not any(p['Distance'] == current_route_start_miles and p['Series'] == 'Actual Fuel' for p in chart_data_points):
        chart_data_points.append({'Distance': current_route_start_miles, 'Fuel': start_fuel, 'Series': 'Actual Fuel', 'Event': 'Start', 'Label': 'Route Start'})
    
    # Sort by distance, then by an implicit order for refuel events
    def sort_key(point):
        if point['Event'] == 'Refuel Start': return (point['Distance'], 0)
        if point['Event'] == 'Refuel End': return (point['Distance'], 2)
        return (point['Distance'], 1)

    chart_data_points.sort(key=sort_key)
    
    fuel_level_df = pd.DataFrame(chart_data_points)

    # Context lines (Tank Capacity, Min Buffer, Desired End)
    route_min_dist = current_route_start_miles
    route_max_dist = current_route_end_miles

    context_data = [
        {'Distance': route_min_dist, 'Fuel': truck_tank_cap, 'Series': 'Tank Capacity'},
        {'Distance': route_max_dist, 'Fuel': truck_tank_cap, 'Series': 'Tank Capacity'},
        {'Distance': route_min_dist, 'Fuel': safety_buffer, 'Series': 'Min Buffer'},
        {'Distance': route_max_dist, 'Fuel': safety_buffer, 'Series': 'Min Buffer'},
        # Desired end fuel shown as a point/short line at destination
        {'Distance': route_max_dist - max(1, route_max_dist*0.005) , 'Fuel': desired_end_fuel, 'Series': 'Desired End Fuel'}, # Show it slightly before end for line visibility
        {'Distance': route_max_dist, 'Fuel': desired_end_fuel, 'Series': 'Desired End Fuel'},
    ]
    context_df = pd.DataFrame(context_data)
    
    combined_df = pd.concat([fuel_level_df, context_df])

    # Base chart
    base = alt.Chart(combined_df).encode(
        x=alt.X('Distance:Q', title='Distance (miles)', scale=alt.Scale(zero=False)),
        y=alt.Y('Fuel:Q', title='Fuel Level (Gallons)', scale=alt.Scale(zero=False)),
        color=alt.Color('Series:N', legend=alt.Legend(title="Series"))
    )

    # Line chart for fuel levels and context lines
    line_chart = base.mark_line(point=False).encode(
            detail='Series:N' # Ensures lines are drawn per series
    )
    
    # Points for specific events (e.g., refueling stops)
    event_points = alt.Chart(fuel_level_df[fuel_level_df['Event'].isin(['Refuel Start', 'Refuel End', 'Destination', 'Start']) | (fuel_level_df['Event'].str.contains("Refuel", case=False, na=False))]).mark_point(
        size=100, filled=True
    ).encode(
        x='Distance:Q',
        y='Fuel:Q',
        shape=alt.Shape('Event:N', legend=alt.Legend(title="Events"))
    ).interactive()
    
    # Text labels for refuel amounts - this is a bit tricky with Altair positioning
    # For simplicity, rely on tooltip for refuel amounts for now.

    final_chart = (line_chart + event_points).properties(
        title='Fuel Level Dynamics Along Route',
        height=500
    ).interactive() # Allow zooming and panning

    st.altair_chart(final_chart, use_container_width=True)

    # Map of route with stations
    st.subheader("Route Map with Stations")

    # Ensure pois_used is not empty before proceeding
    if pois_used:
        # Create a DataFrame from pois_used to easily get mean coordinates for map centering
        map_data = pd.DataFrame(pois_used)


        # Option to use center of US instead
        us_center_lat = 39.8283
        us_center_lon = -88.5795

        # Use US center if requested or if mean coordinates are invalid and satellite view is selected
        m = folium.Map(location=[us_center_lat, us_center_lon], zoom_start=4, control_scale=True)

        # Ensure price_min and price_max are available.
        price_min = 2.5
        price_max = 4.5

        # Assuming price_min and price_max are calculated elsewhere before this block.
        # Let's add a check for their existence and validity.
        if 'price_min' not in locals() or 'price_max' not in locals() or price_min is None or price_max is None:
            st.error("Price range (price_min, price_max) not calculated. Cannot color stations.")
            # Set defaults or handle error appropriately
            price_min = 2.5 # Default fallback
            price_max = 4.5 # Default fallback
            st.info(f"Using default price range: ${price_min:.2f} to ${price_max:.2f}")


        # Now iterate through pois_used to add markers to the map with conditional coloring
        for i, poi in enumerate(pois_used):

            latitude = poi.get('latitude')
            longitude = poi.get('longitude')
            name = poi.get('name', 'Unnamed Location')
            price = poi.get('price_per_gallon', 0)
            is_station = poi.get('is_station', False)
            location_miles = poi.get('location_miles', 0)
            stoppage_cost = poi.get('stoppage_cost', 0) # Include stoppage cost if available
            

            # Skip if coordinates are invalid
            if latitude is None or longitude is None or pd.isna(latitude) or pd.isna(longitude):
                st.warning(f"Skipping marker for '{name}' due to invalid coordinates: ({latitude}, {longitude})")
                continue # Skip this POI and go to the next one

            # Determine color based on whether it's a station and its price
            if is_station:
                # Apply gradient for stations
                # Ensure price_max > price_min to avoid division by zero or issues with uniform prices
                if price_max is not None and price_min is not None and price_max > price_min:
                    try:
                        norm = (price - price_min) / (price_max - price_min)
                        # Clamp norm to [0, 1] to handle prices slightly outside the min/max range
                        norm = max(0, min(1, norm))
                        color = matplotlib.colors.rgb2hex(plt.cm.inferno(norm))
                    except Exception as e:
                        st.warning(f"Error calculating color for station '{name}' (Price: {price:.2f}): {e}. Using gray.")
                        color = 'gray' # Default color on error
                else:
                    # Default color if price range is zero or data is missing
                    color = 'gray'
                popup_text = f"<b>{name}</b><br>Price: ${price:.2f}/Gal<br>Distance: {location_miles:.2f} miles<br>Stoppage Cost: ${stoppage_cost:.2f}"
                marker_color = color
                fill_color = color
                marker_radius = 5
                marker_icon = None # Use default circle marker

            else:
                # Use a different color and potentially icon for non-station points (start/end)
                if poi.get('id') == 'start':
                    color = 'green'
                    popup_text = f"<b>Start: {name}</b><br>Distance: {location_miles:.2f} miles"
                    marker_icon = 'play' # Example icon (requires Font Awesome)
                    marker_radius = 7
                elif poi.get('id') == 'end':
                    color = 'red'
                    popup_text = f"<b>End: {name}</b><br>Distance: {location_miles:.2f} miles"
                    marker_icon = 'flag-checkered' # Example icon (requires Font Awesome)
                    marker_radius = 7
                else:
                    # Default for any other non-station POI
                    color = 'purple'
                    popup_text = f"<b>{name}</b><br>Distance: {location_miles:.2f} miles"
                    marker_icon = None
                    marker_radius = 6

                marker_color = color
                fill_color = color # Use the same color for fill


            # Add marker to the map
            if marker_icon:
                # Use a Marker with an icon for Start/End
                folium.Marker(
                    location=(latitude, longitude),
                    popup=popup_text,
                    icon=folium.Icon(color=marker_color, icon=marker_icon, prefix='fa') # Assuming Font Awesome icons
                ).add_to(m)
            else:
                # Use a CircleMarker for stations or other points without specific icons
                folium.CircleMarker(
                    location=(latitude, longitude),
                    radius=marker_radius,
                    popup=popup_text,
                    color=marker_color,
                    fill=True,
                    fill_color=fill_color,
                    fill_opacity=0.7,
                    tooltip=(f"Store: {name}<br>"
                            f"Price: ${price:.2f}<br>"
                            f"Distance from Start: {location_miles:.1f} mi")
                ).add_to(m)


                # Highlight the actual refueling stops from the optimization solution
                for stop in solution_data['stops']:
                    # Use a more prominent marker for optimized refueling stops
                    folium.Marker(
                        location=(stop['station_latitude'], stop['station_longitude']),
                        popup=f"""<b>⛽ REFUEL HERE: {stop['station_name']}</b><br>
                                  Price: ${stop['price_per_gallon']:.3f}/Gal<br>
                                  Location: {stop['location_miles']:.2f} miles<br>
                                  <hr>
                                  Arrival Fuel: {stop['fuel_at_arrival_gal']:.2f} Gal<br>
                                  <b>Purchase: {stop['fuel_purchased_gal']:.2f} Gal</b><br>
                                  Departure Fuel: {stop['fuel_after_purchase_gal']:.2f} Gal<br>
                                  <b>Cost: ${stop['cost_for_stop']:.2f}</b>""",
                        icon=folium.Icon(color='blue', icon='gas-pump', prefix='fa')
                    ).add_to(m)
                    
                    # Add a circle to make it even more visible
                    folium.CircleMarker(
                        location=(stop['station_latitude'], stop['station_longitude']),
                        radius=7,
                        color='green',
                        fill=True,
                        fill_color='blue',
                        fill_opacity=0.3,
                        weight=2
                    ).add_to(m)
                    
        # Display map
        st_data = st_folium(m, width=900, height=600)

    else:
        st.info("No points of interest available to display on the map.")


