import utils
import pandas as pd
import osmnx as ox
import networkx as nx
import zipfile
from glob import glob
import gzip
import os
import shutil

# Define paths for required directories and create them if they do not exist
worldpop_path = 'data/worldpop/'
os.makedirs(worldpop_path, exist_ok=True)
output_path = 'output/'
os.makedirs(output_path, exist_ok=True)
gtfs_path = 'data/GTFS/'
os.makedirs(gtfs_path, exist_ok=True)

# Iterate through the list of cities sorted by size
for place_name, size, country in sorted(utils.place_list, key=lambda x: x[1]):
    print(place_name)

    # Check if the final graph already exists, skip if it does
    path = f'{output_path}{place_name} - walk & transit.graphml.gz'
    if os.path.isfile(path):
        print("Already done")
        continue

    # Check if the walking network is available, skip if missing
    path = f"{output_path}{place_name}.graphml.gz"
    if not os.path.isfile(path):
        print("Missing walk network")
        continue

    # Load the walking network graph from the compressed file
    G_walk = ox.load_graphml(path)

    # Create empty directed multigraphs for walking and transit networks
    G_walk_transport = nx.MultiDiGraph()
    G_walk_transport.graph['crs'] = "epsg:4326"
    G_transport = nx.MultiDiGraph()
    G_transport.graph['crs'] = "epsg:4326"

    # Define walking speed constants
    WALKING_SPEED_KMH = 5
    WALKING_SPEED_MPS = WALKING_SPEED_KMH * 1000 / 3600  # Convert km/h to m/s

    # Retrieve GTFS zip files for the city
    zip_file_paths = glob(f"{gtfs_path}*{place_name}*.zip")
    if len(zip_file_paths) == 0:
        print("NO GTFS!!")
        continue

    # Initialize empty dataframes to store combined GTFS data
    stops_df_combined = pd.DataFrame()
    stop_times_df_combined = pd.DataFrame()
    trips_df_combined = pd.DataFrame()

    # Load GTFS data from each zip archive
    for zip_file_path in zip_file_paths:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            print(zip_file_path)
            print(zip_ref.namelist())

            # Extract file paths for GTFS datasets
            path_stops = [element for element in zip_ref.namelist() if 'stops.txt' == element.split("/")[-1]][0]
            path_stop_times = [element for element in zip_ref.namelist() if 'stop_times.txt' == element.split("/")[-1]][
                0]
            path_trips = [element for element in zip_ref.namelist() if 'trips.txt' == element.split("/")[-1]][0]


            # Function to read CSV with normalized column names
            def read_csv_with_normalized_columns(file, usecols=None, dtype=None, **kwargs):
                df = pd.read_csv(file, dtype=dtype, **kwargs)
                df.columns = df.columns.str.strip()  # Remove spaces from column names
                if usecols is not None:
                    df = df[[col for col in usecols if col in df.columns]]
                return df


            # Define data types and columns for each GTFS dataset
            dtype_stops = {'stop_id': str, 'stop_lat': str, 'stop_lon': str}
            dtype_stop_times = {'trip_id': str, 'arrival_time': str, 'departure_time': str, 'stop_id': str,
                                'stop_sequence': float}
            dtype_trips = {'route_id': str, 'service_id': str, 'trip_id': str}
            usecols_stops = ['stop_id', 'stop_lat', 'stop_lon']
            usecols_stop_times = ['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence']
            usecols_trips = ['route_id', 'service_id', 'trip_id']

            # Load and concatenate GTFS data
            stops_df = read_csv_with_normalized_columns(zip_ref.open(path_stops), dtype=dtype_stops,
                                                        usecols=usecols_stops, on_bad_lines='skip')
            stop_times_df = read_csv_with_normalized_columns(zip_ref.open(path_stop_times), dtype=dtype_stop_times,
                                                             usecols=usecols_stop_times, on_bad_lines='skip')
            trips_df = read_csv_with_normalized_columns(zip_ref.open(path_trips), dtype=dtype_trips,
                                                        usecols=usecols_trips, on_bad_lines='skip')

            stops_df_combined = pd.concat([stops_df_combined, stops_df], ignore_index=True)
            stop_times_df_combined = pd.concat([stop_times_df_combined, stop_times_df], ignore_index=True)
            trips_df_combined = pd.concat([trips_df_combined, trips_df], ignore_index=True)

    # Final GTFS dataframes
    print("Load the GTFS data")
    stops_df = stops_df_combined
    stop_times_df = stop_times_df_combined
    trips_df = trips_df_combined

    # Convert the walking graph's node coordinates to numeric values
    for node, data in G_walk.nodes(data=True):
        data['x'] = float(data['x'])
        data['y'] = float(data['y'])
        G_walk_transport.add_node(node, x=data['x'], y=data['y'])

    # Convert edge lengths to travel time in seconds
    count_edge = 0
    for u, v, data in G_walk.edges(data=True):
        distance_m = float(data.get('length', 0))  # Edge length in meters
        travel_time_s = distance_m / WALKING_SPEED_MPS  # Convert to seconds
        data['weight'] = travel_time_s
        data['time_travel'] = travel_time_s
        G_walk_transport.add_edge(u, v, key=count_edge, weight=travel_time_s, type="walk", route_id="")
        count_edge += 1

    # Save the final graph as a compressed GraphML file
    print("Saving the final graph")
    graphml_filename = f'{output_path}{place_name} - walk & transit.graphml'
    gzip_filename = f'{output_path}{place_name} - walk & transit.graphml.gz'
    ox.save_graphml(G_walk_transport, graphml_filename)

    # Compress the GraphML file using gzip
    with open(graphml_filename, 'rb') as f_in:
        with gzip.open(gzip_filename, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Remove the uncompressed GraphML file to save space
    os.remove(graphml_filename)
