# Necessary libraries
# You might need to install PuLP: pip install pulp
# You might need to install pandas & openpyxl: pip install pandas openpyxl
import pulp
import math
import pandas as pd

# --- 1. Data Structures (Using Gallons and Miles) ---
class Truck:
    def __init__(self, name, tank_capacity_gal, avg_consumption_gal_per_mile):
        self.name = name
        self.tank_capacity_gal = tank_capacity_gal
        self.avg_consumption_gal_per_mile = avg_consumption_gal_per_mile

class FuelStation: # Represents actual refueling stations
    def __init__(self, id, name, location_miles, price_per_gallon):
        self.id = str(id) 
        self.name = name
        self.location_miles = location_miles
        self.price_per_gallon = price_per_gallon

    def __repr__(self):
        return (f"FuelStation(id='{self.id}', name='{self.name}', "
                f"location_miles={self.location_miles:.2f}, price_per_gallon={self.price_per_gallon:.3f})")

class Route:
    def __init__(self, name, start_location_miles, end_location_miles, station_ids_in_order):
        self.name = name
        self.start_location_miles = start_location_miles
        self.end_location_miles = end_location_miles
        self.station_ids_in_order = [str(sid) for sid in station_ids_in_order]

# --- Helper function to load stations and identify endpoint from Excel ---
def load_stations_and_endpoint_from_excel(file_path, sheet_name=0, endpoint_store_no="-1"):
    """
    Loads fuel station data and identifies a specific endpoint (e.g., Cheshire)
    from an Excel file. Distances are expected in miles, prices per gallon.

    Args:
        file_path (str): The path to the Excel file.
        sheet_name (str or int): The sheet name or index to read from.
        endpoint_store_no (str): The 'Store No' value that identifies the endpoint row.

    Returns:
        tuple: (list_of_fuel_stations, endpoint_details_dict)
               endpoint_details_dict contains {'name': str, 'location_miles': float} or None if not found.
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Successfully read Excel file. Columns found: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"Error: Excel file not found at {file_path}")
        return [], None
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return [], None

    fuel_stations = []
    endpoint_details = None
    
    # --- Column Name Assumptions (ADJUST AS NEEDED based on your Excel header) ---
    id_col = 'Store No'
    name_col_city = 'City' # Used for station name and endpoint name
    location_col_miles_excel = 'Distance from Fresno (miles)' # Already in miles
    price_col_excel = 'Best Discounted Price' # IMPORTANT: Assumed to be PRICE PER GALLON for fuel stations

    required_cols = [id_col, name_col_city, location_col_miles_excel] # Price is not required for endpoint row
    if price_col_excel not in df.columns:
        print(f"Warning: Price column '{price_col_excel}' not found. Stations will not have price data.")
        # If price column is critical for stations, you might want to make this an error:
        # required_cols.append(price_col_excel)


    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Required column(s) {missing_cols} not found in Excel sheet.")
        print(f"Please ensure '{file_path}' contains these columns with exact names.")
        return [], None

    for index, row in df.iterrows():
        try:
            station_id_str = str(row[id_col]).strip()
            city_name = str(row[name_col_city]).strip()
            
            current_row_location_miles = pd.to_numeric(row[location_col_miles_excel], errors='coerce')
            if pd.isna(current_row_location_miles) or current_row_location_miles < 0:
                print(f"Warning: Skipping row {index+2} (ID: {station_id_str}) due to invalid or negative distance: '{row[location_col_miles_excel]}'")
                continue

            if station_id_str == endpoint_store_no:
                if endpoint_details is not None:
                    print(f"Warning: Multiple endpoint entries found with Store No '{endpoint_store_no}'. Using the first one encountered: '{endpoint_details['name']}'.")
                else:
                    endpoint_details = {'name': city_name if city_name else "Endpoint", 'location_miles': current_row_location_miles}
                    print(f"Identified endpoint: Name='{endpoint_details['name']}', Location={endpoint_details['location_miles']:.2f} miles (from row {index+2})")
                continue # Don't process endpoint as a fuel station

            # Process as a regular fuel station
            station_name = f"{city_name} (ID: {station_id_str})"
            
            price_per_gallon = 0.0 # Default if price column missing or invalid
            if price_col_excel in df.columns:
                # IMPORTANT ASSUMPTION: Price in Excel is per gallon.
                # If your Excel price is per liter, you must convert it here:
                # price_per_liter = pd.to_numeric(row[price_col_excel], errors='coerce')
                # price_per_gallon = price_per_liter * 3.78541 # Conversion factor
                price_val = pd.to_numeric(row[price_col_excel], errors='coerce')
                if pd.isna(price_val) or price_val <= 0:
                    print(f"Warning: Station ID {station_id_str} (row {index+2}) has invalid or non-positive price: '{row[price_col_excel]}'. Assuming $0.0/Gal or station might be unusable if price is critical.")
                    # Depending on your needs, you might skip stations with invalid prices:
                    # continue 
                else:
                    price_per_gallon = price_val
            
            fuel_stations.append(FuelStation(id=station_id_str, name=station_name, location_miles=current_row_location_miles, price_per_gallon=price_per_gallon))

        except KeyError as e:
            print(f"Warning: Missing expected column for row {index+2}. Error: {e}. Skipping this row.")
            continue
        except Exception as e: # Catch any other unexpected error during row processing
            print(f"Warning: Error processing row {index+2} (ID: {station_id_str}). Error: {e}. Skipping this row.")
            continue
            
    fuel_stations.sort(key=lambda s: s.location_miles)
    # Validation print moved to main section for clarity after loading
    return fuel_stations, endpoint_details

# --- 2. Helper Functions ---
def get_station_by_id(station_id, all_stations):
    target_station_id = str(station_id)
    for station in all_stations:
        if station.id == target_station_id:
            return station
    return None

# --- 3. Core Optimization Function (Using Gallons and Miles) ---
# (This function remains largely the same as in the previous Gallons/Miles version)
def optimize_fuel_stops(
    truck: Truck,
    route: Route,
    all_stations_data: list, # This is the list of FuelStation objects
    start_fuel_gal: float,
    desired_end_fuel_gal: float,
    min_buffer_fuel_gal: float,
    max_refueling_stops: int):

    pois = []
    # Add start POI
    pois.append({'id': 'start', 'name': 'Start Route', 'location_miles': route.start_location_miles, 'price_per_gallon': 0, 'is_station': False})
    
    route_specific_stations_objects = []
    if all_stations_data: # Ensure there are stations to iterate through
        for station_id_on_route in route.station_ids_in_order:
            station_obj = get_station_by_id(station_id_on_route, all_stations_data)
            if station_obj:
                # Crucial check: only include stations strictly between start and end, or at start if it's a station.
                # End location itself is not a refueling spot unless explicitly modeled as such (which it isn't here).
                if route.start_location_miles <= station_obj.location_miles < route.end_location_miles:
                    route_specific_stations_objects.append(station_obj)
                # If a station is exactly AT the end_location_miles, it won't be added as a refueling stop here.
                # This is generally correct as you arrive at the destination, not refuel there as part of the trip segment.
                elif station_obj.location_miles == route.end_location_miles:
                     print(f"Info: Station {station_obj.name} (ID: {station_obj.id}) is at the exact destination location ({route.end_location_miles:.2f} miles) and will not be considered a refueling stop on the way.")
                elif station_obj.location_miles > route.end_location_miles :
                     print(f"Info: Station {station_obj.name} (ID: {station_obj.id}) at {station_obj.location_miles:.2f} miles is beyond the route's end ({route.end_location_miles:.2f} miles) and will be excluded.")
                # else: (station is before start, also excluded by station_ids_in_order logic usually)

            else: # station_obj is None
                print(f"Warning: Station ID '{station_id_on_route}' defined in route '{route.name}' not found in the provided list of all_stations_data.")
    else:
        print("Info: No fuel stations provided to select from for the route.")
            
    route_specific_stations_objects.sort(key=lambda s: s.location_miles)

    for station in route_specific_stations_objects:
        pois.append({
            'id': station.id, 
            'name': station.name, 
            'location_miles': station.location_miles, 
            'price_per_gallon': station.price_per_gallon, 
            'is_station': True
        })

    # Add end POI using the route's defined end location
    pois.append({'id': 'end', 'name': 'End Route (Destination)', 'location_miles': route.end_location_miles, 'price_per_gallon': 0, 'is_station': False})
    
    # Clean up POIs: Remove duplicates by location, prioritizing stations if conflicts. Start/End are special.
    unique_pois_dict = {}
    for p in sorted(pois, key=lambda x: (x['location_miles'], not x['is_station'])): # Sort to process stations first for a given location
        loc = p['location_miles']
        if loc not in unique_pois_dict:
            unique_pois_dict[loc] = p
        elif p['is_station'] and not unique_pois_dict[loc]['is_station']: # if current is station and stored is not
            unique_pois_dict[loc] = p
        elif (p['id'] == 'start' or p['id'] == 'end') and not unique_pois_dict[loc]['is_station']: # if current is start/end and stored is not station
             unique_pois_dict[loc] = p


    pois = sorted(list(unique_pois_dict.values()), key=lambda p_item: p_item['location_miles'])
    
    num_pois = len(pois)
    if num_pois < 2: # Should have at least Start and End
        print("Error: Route must have at least a start and end point after processing POIs. Check POI list:")
        for i, p_item in enumerate(pois): print(f"  POI {i}: {p_item}")
        return None
    
    actual_station_poi_indices = [i for i, p_item in enumerate(pois) if p_item['is_station']]
    print(f"Number of Points of Interest (POIs) for optimization: {num_pois}")
    print(f"Number of actual fuel stations among POIs: {len(actual_station_poi_indices)}")


    prob = pulp.LpProblem(f"FuelOptimization_{truck.name}_{route.name}", pulp.LpMinimize)

    fuel_purchased_gal = pulp.LpVariable.dicts("FuelPurchasedGal", range(num_pois), lowBound=0, cat='Continuous')
    fuel_at_arrival_gal = pulp.LpVariable.dicts("FuelAtArrivalGal", range(num_pois), lowBound=min_buffer_fuel_gal, upBound=truck.tank_capacity_gal, cat='Continuous')
    fuel_after_purchase_gal = pulp.LpVariable.dicts("FuelAfterPurchaseGal", range(num_pois), lowBound=min_buffer_fuel_gal, upBound=truck.tank_capacity_gal, cat='Continuous')

    stop_decision = {}
    if actual_station_poi_indices:
        stop_decision = pulp.LpVariable.dicts("StopDecision", actual_station_poi_indices, cat='Binary')
    else:
        print("Info: No actual fuel stations identified on the route for optimization variables.")

    total_cost = pulp.LpAffineExpression()
    for i in range(num_pois):
        if pois[i]['is_station']: # This index i must be in actual_station_poi_indices
            if i not in actual_station_poi_indices: # Should not happen due to how actual_station_poi_indices is built
                 print(f"CRITICAL LOGIC ERROR: POI {i} ('{pois[i]['name']}') is_station=True but not in actual_station_poi_indices.")
                 continue # or raise error
            total_cost += fuel_purchased_gal[i] * pois[i]['price_per_gallon']
    prob += total_cost, "TotalFuelCost"

    # Initial Fuel Constraint (at POI 0 - Start)
    if not (pois[0]['id'] == 'start' and pois[0]['location_miles'] == route.start_location_miles):
        print(f"Error: First POI is not the route start. POI[0]: {pois[0]}. Route Start: {route.start_location_miles}")
        return None
    prob += fuel_at_arrival_gal[0] == start_fuel_gal, "InitialFuelConstraint"
    prob += fuel_after_purchase_gal[0] == fuel_at_arrival_gal[0], "NoPurchaseAtStartConstraint"
    prob += fuel_purchased_gal[0] == 0, f"ZeroPurchaseAtStartPOI"

    for i in range(num_pois):
        if pois[i]['is_station']: # This POI i must be an actual station
            if i not in actual_station_poi_indices or i not in stop_decision:
                 print(f"Critical Error during constraint setup: POI {i} ({pois[i]['name']}) is station but has inconsistent setup for stop_decision. Check indices.")
                 return None # Problem with how stop_decision keys (actual_station_poi_indices) were made
            prob += fuel_after_purchase_gal[i] == fuel_at_arrival_gal[i] + fuel_purchased_gal[i], f"FuelBalanceAfterPurchase_{pois[i]['id']}"
            prob += fuel_purchased_gal[i] <= truck.tank_capacity_gal * stop_decision[i], f"LinkPurchaseToStopDecisionUpper_{pois[i]['id']}"
            prob += fuel_purchased_gal[i] >= 0.01 * stop_decision[i], f"LinkPurchaseToStopDecisionLower_{pois[i]['id']}" 
            prob += fuel_purchased_gal[i] <= truck.tank_capacity_gal - fuel_at_arrival_gal[i], f"PurchaseCapacityLimit_{pois[i]['id']}"
        else: # Not a refueling station (Start, End, or other non-fuel POI)
            prob += fuel_after_purchase_gal[i] == fuel_at_arrival_gal[i], f"NoPurchaseNonStation_{pois[i]['id']}"
            prob += fuel_purchased_gal[i] == 0, f"ZeroPurchaseNonStation_{pois[i]['id']}"

        # Fuel flow to next POI
        if i < num_pois - 1: # If not the last POI
            dist_to_next_miles = pois[i+1]['location_miles'] - pois[i]['location_miles']
            if dist_to_next_miles < 0: # Should not happen if POIs sorted by location_miles
                print(f"Error: Negative distance from POI {i} ('{pois[i]['name']}') to POI {i+1} ('{pois[i+1]['name']}'). Dist: {dist_to_next_miles:.2f} miles. POIs not sorted correctly.")
                return None
            
            fuel_needed_for_segment_gal = dist_to_next_miles * truck.avg_consumption_gal_per_mile
            prob += fuel_at_arrival_gal[i+1] == fuel_after_purchase_gal[i] - fuel_needed_for_segment_gal, f"FuelFlow_{pois[i]['id']}_to_{pois[i+1]['id']}"
            # Ensure enough fuel to make it to the next stop AND arrive with at least the buffer
            prob += fuel_after_purchase_gal[i] >= fuel_needed_for_segment_gal + min_buffer_fuel_gal, f"SufficientFuelForSegment_{pois[i]['id']}_to_{pois[i+1]['id']}"

    # Desired End Fuel Constraint (at the last POI - End)
    if not (pois[num_pois-1]['id'] == 'end' and pois[num_pois-1]['location_miles'] == route.end_location_miles):
        print(f"Error: Last POI is not the route end. POI[{num_pois-1}]: {pois[num_pois-1]}. Route End: {route.end_location_miles}")
        return None
    prob += fuel_at_arrival_gal[num_pois-1] >= desired_end_fuel_gal, "DesiredEndFuelConstraint"

    # Max Stops Constraint
    if actual_station_poi_indices and stop_decision: # If there are stations and decision variables for them
        prob += pulp.lpSum(stop_decision[idx] for idx in actual_station_poi_indices) <= max_refueling_stops, "MaxStopsConstraint"
    elif not actual_station_poi_indices and max_refueling_stops >=0 :
        print(f"Info: Max stops constraint ({max_refueling_stops}) is set, but no fuel stations are part of the optimization POIs.")
    
    # Solve
    solver_to_use = pulp.PULP_CBC_CMD(msg=0)
    try:
        status = prob.solve(solver_to_use)
    except pulp.apis.core.PulpSolverError as e:
        print(f"PulpSolverError occurred: {e}. Ensure solver (CBC) is installed and in PATH.")
        return None

    # Process and return results
    if pulp.LpStatus[status] == "Optimal":
        print("\nOptimal solution found!")
        print(f"Total Fuel Cost: ${pulp.value(prob.objective):.2f}")
        
        stops_made_val = 0
        if actual_station_poi_indices and stop_decision: # Check if stop_decision was populated
            stops_made_val = sum(pulp.value(stop_decision[idx]) for idx in actual_station_poi_indices if idx in stop_decision and pulp.value(stop_decision[idx]) is not None)

        solution = {
            'total_cost': pulp.value(prob.objective),
            'truck': truck.name,
            'route': route.name,
            'stops_made_count': stops_made_val,
            'stops': []
        }
        
        print(f"Number of refueling stops made: {solution['stops_made_count']:.0f} (Max allowed: {max_refueling_stops})")
        print("\nFueling Plan (US Gallons, Miles):")
        header = f"{'Location':<40} | {'Arrival Fuel (Gal)':<20} | {'Purchased Fuel (Gal)':<22} | {'Depart Fuel (Gal)':<19} | {'Cost ($)':<10} | {'Price ($/Gal)':<12}"
        print(header)
        print("-" * len(header))

        for i in range(num_pois):
            poi_name = pois[i]['name']
            # Ensure values are not None before formatting
            arr_fuel_val = pulp.value(fuel_at_arrival_gal[i])
            arr_fuel = arr_fuel_val if arr_fuel_val is not None else 0.0
            
            pur_fuel = 0.0
            if pois[i]['is_station']:
                var_value = pulp.value(fuel_purchased_gal[i])
                if var_value is not None: pur_fuel = var_value
                else: print(f"Warning: fuel_purchased_gal[{i}] value is None for station {poi_name}. Assuming 0.")
            
            dep_fuel_val = pulp.value(fuel_after_purchase_gal[i])
            dep_fuel = dep_fuel_val if dep_fuel_val is not None else 0.0

            cost_at_stop = pur_fuel * pois[i]['price_per_gallon'] if pois[i]['is_station'] and pur_fuel > 0.001 else 0.0
            price_at_stop_str = f"{pois[i]['price_per_gallon']:.3f}" if pois[i]['is_station'] else "-"
            
            is_stop_decision_one = False
            if pois[i]['is_station'] and i in stop_decision: # Check if stop_decision[i] exists
                stop_decision_val = pulp.value(stop_decision[i])
                if stop_decision_val is not None and stop_decision_val == 1:
                    is_stop_decision_one = True
            
            if is_stop_decision_one and pur_fuel > 0.001 : # A stop is counted if decision is 1 and fuel is bought
                 solution['stops'].append({
                    'station_id': pois[i]['id'],
                    'station_name': poi_name,
                    'location_miles': pois[i]['location_miles'],
                    'fuel_at_arrival_gal': round(arr_fuel,2),
                    'fuel_purchased_gal': round(pur_fuel,2),
                    'fuel_after_purchase_gal': round(dep_fuel,2),
                    'price_per_gallon': pois[i]['price_per_gallon'],
                    'cost_for_stop': round(cost_at_stop,2)
                })
            print(f"{poi_name:<40} | {arr_fuel:<20.2f} | {pur_fuel:<22.2f} | {dep_fuel:<19.2f} | {cost_at_stop:<10.2f} | {price_at_stop_str:<12}")
        print("-" * len(header))
        return solution
    else:
        print(f"Optimization failed. Status: {pulp.LpStatus[status]}")
        debug_file = f"fuel_problem_debug_{truck.name}_{route.name.replace(' ','_')}.lp"
        try: prob.writeLP(debug_file); print(f"Problem written to {debug_file}.")
        except Exception as e_lp: print(f"Could not write LP file: {e_lp}")
        return None

# --- 4. Example Usage (Using Gallons and Miles, Endpoint from Excel) ---
if __name__ == '__main__':
    excel_file_path = 'fresno_to_cheshire_stations.xlsx' 
    # Load stations AND the specific endpoint (Cheshire with Store No = -1)
    all_fuel_stations, cheshire_endpoint = load_stations_and_endpoint_from_excel(excel_file_path, endpoint_store_no="-1")

    # --- Validation Prints ---
    print(f"\n--- Data Loading Validation ---")
    if cheshire_endpoint:
        print(f"Detected Endpoint: Name='{cheshire_endpoint['name']}', Location={cheshire_endpoint['location_miles']:.2f} miles.")
    else:
        print(f"CRITICAL: Cheshire endpoint (Store No = -1) not found in '{excel_file_path}'. Cannot define route endpoint. Exiting.")
        exit()
        
    print(f"Total actual fuel stations loaded from Excel: {len(all_fuel_stations)}")
    if not all_fuel_stations:
        print(f"Warning: No actual fuel stations were loaded from '{excel_file_path}'. Check data and column names ('{excel_file_path}').")
    # --- End Validation ---

    # --- Truck Definition (Gallons, Gal/Mile) ---
    truck_tank_capacity = 240.0  # Gallons
    truck_mpg = 7.0              # Miles per Gallon
    truck_consumption_gpm = 1.0 / truck_mpg if truck_mpg > 0 else 0.2 

    my_truck_imperial = Truck(name="FreightlinerImperial", 
                              tank_capacity_gal=truck_tank_capacity, 
                              avg_consumption_gal_per_mile=truck_consumption_gpm)

    # --- Route Definition (Miles) using endpoint from Excel ---
    fresno_origin_miles = 0.0 
    # Use Cheshire's location from Excel as the end_location_miles
    route_end_miles = cheshire_endpoint['location_miles'] 
    
    # Select stations that are on the path to Cheshire
    # Stations strictly before the endpoint are candidates.
    station_ids_for_route = [s.id for s in all_fuel_stations if s.location_miles < route_end_miles and s.location_miles >= fresno_origin_miles]
    
    # Also include stations AT the origin, if any (e.g. a truck stop at Fresno)
    # station_ids_at_origin = [s.id for s in all_fuel_stations if s.location_miles == fresno_origin_miles]
    # station_ids_for_route = list(set(station_ids_for_route + station_ids_at_origin)) # Ensure uniqueness

    fresno_to_cheshire_rt = Route(
        name=f"Fresno_to_{cheshire_endpoint['name']}",
        start_location_miles=fresno_origin_miles,
        end_location_miles=route_end_miles, 
        station_ids_in_order=station_ids_for_route # Optimizer will sort them by distance later
    )
    print(f"\nDefined Route: {fresno_to_cheshire_rt.name} from {fresno_to_cheshire_rt.start_location_miles:.2f} miles to {fresno_to_cheshire_rt.end_location_miles:.2f} miles (Destination: {cheshire_endpoint['name']})")
    print(f"Number of fuel stations selected for this route (before destination): {len(station_ids_for_route)}")
    if not station_ids_for_route :
        print("Warning: No fuel stations identified on the route before the destination.")


    # --- Optimization Parameters (Gallons) ---
    start_fuel = 100.0           # Gallons
    desired_end_fuel = 150.0     # Gallons (desired fuel at destination - Cheshire)
    safety_buffer_fuel = 50.0   # Gallons (minimum fuel buffer)
    max_stops = 7               # Max number of refueling stops

    print(f"\n--- Optimization Setup ---")
    print(f"Truck: {my_truck_imperial.name}, Capacity={my_truck_imperial.tank_capacity_gal:.1f} Gal, Consumption={my_truck_imperial.avg_consumption_gal_per_mile:.4f} Gal/Mile ({1/my_truck_imperial.avg_consumption_gal_per_mile:.1f} MPG)")
    print(f"Fuel Parameters: Start={start_fuel:.1f} Gal, Desired End at {cheshire_endpoint['name']}={desired_end_fuel:.1f} Gal, Safety Buffer={safety_buffer_fuel:.1f} Gal")
    print(f"Constraints: Max Refueling Stops Allowed={max_stops}")

    optimization_result = optimize_fuel_stops(
        truck=my_truck_imperial,
        route=fresno_to_cheshire_rt,
        all_stations_data=all_fuel_stations, # Pass the list of FuelStation objects
        start_fuel_gal=start_fuel,
        desired_end_fuel_gal=desired_end_fuel,
        min_buffer_fuel_gal=safety_buffer_fuel,
        max_refueling_stops=max_stops
    )

    if optimization_result:
        print(f"\nOptimization successful for {fresno_to_cheshire_rt.name}.")
    else:
        print(f"\nNo optimal solution found for {fresno_to_cheshire_rt.name}.")
        fuel_needed_approx = (fresno_to_cheshire_rt.end_location_miles * my_truck_imperial.avg_consumption_gal_per_mile) 
        print(f"  Approx fuel needed for route distance: {fuel_needed_approx:.2f} Gallons")
        max_range_on_start_fuel_approx = (start_fuel - safety_buffer_fuel) / my_truck_imperial.avg_consumption_gal_per_mile if my_truck_imperial.avg_consumption_gal_per_mile > 0 else 0
        print(f"  Approx max range on start fuel (with buffer): {max_range_on_start_fuel_approx:.2f} Miles")

    print("\n--- Script finished ---")