from shapely.geometry import MultiPoint
from utils import (
    categories,
    get_queryable_tags,
    get_extended_bebop_from_graph,
    filter_pois_by_tags,
    filter_pois_by_polygon,
    place_list, calculate_area_in_square_km,
    country_tif_urls
)
import math
import numpy as np
import concurrent.futures
import threading
import os
import json
import gzip
import shutil
import gc
import concurrent.futures
import pandas as pd
import geopandas as gpd
import rioxarray
import networkx as nx
import osmnx as ox
from shapely import wkt
from tqdm import tqdm
from more_itertools import chunked
import requests

worldpop_path = f'data/worldpop/'
os.makedirs(worldpop_path, exist_ok=True)
output_path =  f'output/'
os.makedirs(output_path, exist_ok=True)


# Save checkpoints during processing
CHECKPOINT = False

def load_checkpoint(filename):
    """
    Loads a checkpoint from a JSON file.

    Parameters:
    - filename: Path to the checkpoint file.

    Returns:
    - Dictionary containing the checkpoint data if the file exists, otherwise an empty dictionary.
    """
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint_async(result, filename):
    """
    Saves a checkpoint asynchronously.

    Parameters:
    - result: Data to be saved in the checkpoint.
    - filename: Path to the checkpoint file.

    This function spawns a new thread to handle the file writing operation,
    preventing blocking in the main execution flow.
    """

    def async_save(result, filename):
        with open(filename, "w") as f:
            json.dump(result, f, indent=4)

    threading.Thread(target=async_save, args=(result, filename)).start()



#If the node is "close" to one for which proximity, density, and entropy have already been computed, it will inherit the values from the precomputed node.
#If I use caching, I might get different results between two runs on the same city if I change the number of parallel executions or the batch size. This happens because different settings can alter the execution order of the nodes, which in turn affects whether and how the method retrieves data from the cache.
#The use of checkpoints resets the cache.


# Use maximum distance to consider two nodes as "close" (in meters)
MAX_DISTANCE = 50
# Cache to store already processed nodes and their results
node_cache = {
    "coordinates": np.empty((0, 2)),  # NumPy array to store coordinates (latitude, longitude)
    "results": []  # List to store results (proximity, density, entropy )
}

# Earth radius meters
EARTH_RADIUS = 6371000
def haversine_np(lat1, lon1, lat2, lon2):
    """
    Computes the approximate Haversine distance between two points (latitude and longitude).

    Parameters:
    - lat1, lon1: coordinates of the first point (in degrees)
    - lat2, lon2: coordinates of the second point (in degrees)

    Returns:
    - Distance in meters.
    """
    # Convert latitudes and longitudes from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Differences between coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Distance in meters
    distance = EARTH_RADIUS * c
    return distance

def compute_with_cache(G, pois, node, travel_speed, max_travel_time):
    """
    Computes or reuses results for a specific node using a cache.

    Parameters:
    - G: city graph.
    - pois: points of interest.
    - node: current node.
    - travel_speed: travel speed.
    - current_max_travel_time: maximum travel time.

    Returns:
    - Computed or reused results (proximity, density, entropy).
    """

    # Get the coordinates of the current node
    current_coords = np.array([float(G.nodes[node]['y']), float(G.nodes[node]['x'])])  # lat, lon

    # If there are already processed nodes, compute distances vectorized
    if node_cache["coordinates"].shape[0] > 0:
        # Extract coordinates from the cache
        lat_cache = node_cache["coordinates"][:, 0]
        lon_cache = node_cache["coordinates"][:, 1]

        # Vectorized computation of distances between the current node and all cached nodes
        distances = haversine_np(current_coords[0], current_coords[1], lat_cache, lon_cache)

        # Check if there is a nearby node (within MAX_DISTANCE meters)
        if np.any(distances <= MAX_DISTANCE):
            closest_index = np.argmin(distances)
            # print(f"Reusing result for node {node}, close to cached node (distance: {distances[closest_index]:.2f} meters)")

            return [
                node,  # node
                node_cache["results"][closest_index][1],  # proximity
                node_cache["results"][closest_index][2],  # entropy
                node_cache["results"][closest_index][3],  # density
                node_cache["results"][closest_index][4],  # coordinates
            ]

    # Compute for nodes not present in the cache
    start_node, proximity, density, entropy, (lat, lon) = compute_proximity_density_entropy(
        G, pois, node, travel_speed, max_travel_time
    )

    # Store the result in the cache
    this_result = [start_node, proximity, density, entropy, (lat, lon)]

    if proximity is not None:
        # Add the node and result to the cache
        node_cache["coordinates"] = np.vstack([node_cache["coordinates"], current_coords])  # Add new coordinates
        node_cache["results"].append(this_result)  # Add the corresponding result

    return this_result


def compute_proximity_density_entropy(G, pois, node, travel_speed, max_travel_time):
    """
    Computes proximity, density, and entropy for a given node, considering points of interest (POIs)
    and travel constraints.

    Parameters:
    - G: city graph.
    - pois: dictionary of points of interest categorized by type.
    - node: current node being processed.
    - travel_speed: travel speed in km/h.
    - current_max_travel_time: maximum allowed travel time in minutes.

    Returns:
    - A list containing:
        - start_node: the original node.
        - old_proximity: a dictionary with proximity values for different categories.
        - density: the computed POI density within the isochrone.
        - entropy: entropy of POIs in the area.
        - (lat, lon): coordinates of the node.
    """

    # Processing nodes that are not in the cache
    step = 5
    old_proximity = {}

    # Iteratively compute proximity in increasing time intervals
    for cur_max_travel_time in range(step, max_travel_time + step, step):
        result = compute_proximity(
            G, pois, node, travel_speed, cur_max_travel_time - step + 1, cur_max_travel_time
        )
        start_node, proximity, (lat, lon) = result

        # Cases where the node exists in G but not in the extended graph (not saved in JSON)
        if proximity is None:
            return [start_node, None, None, None, (lat, lon)]

        # Store the first available proximity value for each category
        for category in categories.keys():
            if category not in old_proximity and proximity[category] != MAX_TRAVEL_TIME + 1:
                old_proximity[category] = proximity[category]

        # Stop if all categories have meaningful proximity values
        if max(proximity.values()) != MAX_TRAVEL_TIME + 1:
            break

    # Define a 15-minute travel isochrone
    travel_time = 15  # minutes
    distance = int((travel_time * 60) * travel_speed / 3.6)  # Convert to meters
    subgraph = nx.ego_graph(G, node, radius=distance, distance='length')

    # Extract node coordinates within the isochrone
    node_coords = [(subgraph.nodes[node]['x'], subgraph.nodes[node]['y']) for node in subgraph.nodes()]
    isochrone_15_polygon = MultiPoint(node_coords).convex_hull

    # Count POIs within the 15-minute isochrone for each category
    POI_in_C = {}
    for category, data in categories.items():
        found_pois = filter_pois_by_polygon(pois[category], isochrone_15_polygon)
        POI_in_C[category] = len(found_pois)

    # Compute entropy
    entropy = 0
    total_pois = sum(POI_in_C.values())

    if total_pois > 0:
        for category in categories.keys():
            frac_in_C = POI_in_C[category] / total_pois
            if frac_in_C > 0:
                entropy += frac_in_C * abs(math.log(frac_in_C, len(categories)))
        entropy = abs(entropy)

    # Compute density (POIs per square km)
    area = calculate_area_in_square_km(isochrone_15_polygon)

    if area is None or area < 1:
        return [start_node, None, None, None, (lat, lon)]

    density = total_pois / area

    return [start_node, old_proximity, density, entropy, (lat, lon)]


def compute_proximity(G, pois, start_node, travel_speed=4.8, from_max_travel=1, current_max_travel_time=60):
    """
    Computes the proximity of a node to the nearest points of interest (POIs) for each category.

    Parameters:
    - G: city graph.
    - pois: dictionary of points of interest categorized by type.
    - start_node: the node from which proximity is computed.
    - travel_speed: travel speed in km/h (default: 4.8 km/h, approximately walking speed).
    - from_max_travel: the starting travel time threshold for proximity calculation (default: 1 minute).
    - current_max_travel_time: maximum allowed travel time in minutes in this step

    Returns:
    - A tuple containing:
        - start_node: the original node.
        - proximity: a dictionary with the minimum travel time required to reach a POI for each category.
        - (x, y): coordinates of the node.
    """

    # Retrieve the population at the given node
    population = float(G.nodes[start_node]['worldpop_count'])

    # If the population is zero or undefined, return no proximity data
    if not population > 0:
        return start_node, None, (G.nodes[start_node]['x'], G.nodes[start_node]['y'])

    # Initialize variables
    isochrone_polygons = {}
    proximity = {}
    found_pois_category = {}

    # Filter the graph based on max travel time to reduce processing priority
    max_distance = int((current_max_travel_time * 60) * travel_speed / 3.6)  # Convert travel time to meters
    subG = nx.ego_graph(G, start_node, radius=max_distance, distance='length')

    # Compute the isochrone polygon for the maximum travel time
    node_coords = [(subG.nodes[node]['x'], subG.nodes[node]['y']) for node in subG.nodes()]
    isochrone_polygons[current_max_travel_time] = MultiPoint(node_coords).convex_hull

    for category in categories.keys():
        # Filter POIs within the maximum travel polygon to reduce processing time
        # Note: POIs are already grouped by category
        filtered_pois = filter_pois_by_polygon(pois[category], isochrone_polygons[current_max_travel_time])

        # Initialize proximity with a default value (larger than any possible travel time)
        proximity[category] = MAX_TRAVEL_TIME + 1

        # Compute the minimum travel time to a POI in this category
        for travel_time in range(int(from_max_travel), int(current_max_travel_time) + 1):
            if travel_time not in isochrone_polygons:
                distance = int((travel_time * 60) * travel_speed / 3.6)  # Convert to meters
                subgraph = nx.ego_graph(subG, start_node, radius=distance, distance='length')
                node_coords = [(subgraph.nodes[node]['x'], subgraph.nodes[node]['y']) for node in subgraph.nodes()]
                isochrone_polygons[travel_time] = MultiPoint(node_coords).convex_hull

            found_pois = filter_pois_by_polygon(filtered_pois, isochrone_polygons[travel_time])
            found_pois_category[category] = found_pois

            if len(found_pois) == 0:
                continue

            # If POIs are found, record the travel time and break the loop
            proximity[category] = travel_time
            break

    return start_node, proximity, (subG.nodes[start_node]['x'], subG.nodes[start_node]['y'])



def compute_batch(G, pois, nodes_batch, travel_speed, max_travel_time):
    """
    Computes results in batch for a group of nodes.

    Parameters:
    - G: city graph.
    - pois: points of interest.
    - nodes_batch: batch of nodes to process.
    - travel_speed: movement speed.
    - current_max_travel_time: maximum travel time.

    Returns:
    - A list of results for the nodes in the batch.
    """
    results = []
    for node in nodes_batch:
        result = compute_with_cache(G, pois, str(node), travel_speed, max_travel_time)
        # result = compute_proximity_density_entropy(G, pois, str(node), travel_speed, current_max_travel_time)

        results.append(result)
    return results


#Parameters for configuring the maximum isochrone considered in computation

# Maximum travel time (minutes)
MAX_TRAVEL_TIME = 60
# Travel speed (km/h)
travel_speed = 4.8     # Transport for London recommend 1.33 metres per second (4.8 km/h; 3.0 mph; 4.4 ft/s) in the PTAL methodology.

# Parameters for configuring the computation
# Nodes to process in each batch
BATCH_SIZE = 200
# Number of parallel processes to start (consider CPU count and RAM availability)
MAX_WORKERS = 2


if __name__ == '__main__':
    """
    Main execution script to compute Points of Interest (PoI) accessibility for multiple cities.

    The script:
    - Checks if the city has already been processed.
    - Loads population data from WorldPop.
    - Loads previous checkpoints if available.
    - Builds the city graph from OpenStreetMap.
    - Extends the graph for better isochrone computations.
    - Extracts PoIs from the extended graph.
    - Uses parallel processing to compute accessibility metrics in batches.
    - Saves results periodically and at the end of processing.
    
    
    For each city, up to five files will be generated:

    data/worldpop/{country} – A raster file downloaded from WorldPop, containing population estimates. This file is downloaded only once per country and used for all cities within that nation.
    {place_name}.graphml.gz – The city's road network extracted from OpenStreetMap (OSM).
    {place_name} extended.graphml.gz – An expanded version of the city's bounding box, covering the maximum travel distance reachable within the set time and speed constraints.
    {place_name}.csv.gz – A file containing Points of Interest (PoIs) retrieved from OSM
    
    
     A JSON file storing accessibility data for each intersection. Each entry represents a node with the following structure:
        {
            "29705609": {                               #The OSM intersection ID
                "coordinate": [38.7906608, 9.0150595],  #Intersection lat e lon
                "proximity": {   
                    "mobility": 4,
                    "active_living": 4,
                    "entertainment": 6,
                    "food": 2,
                    "community": 6,
                    "education": 6,
                    "health_and_wellbeing": 6
                },
                "density": 41.355966962356675,
                "entropy": 0.7821541410603423,
                "worldpop_count": 105.809814453125, #Worldpop population estimated in the grid including that interseciton
                "worldpop_code": "38.79083319268429,9.015000165047434"    #Worldpop centroid grid
            },
            ...
        }
        
     se proximity ha meno di 7 elementi, significa che la categoria mancante non è raggiungibile entro il massimo tempo di camminta a quella voleocità.
     se proxity è none significa che la zona non è popolata (in quei casi non abbiamo calcolato la proximity), ne consegue che
             "proximity": null,
            "density": null,
            "entropy": null,
            "worldpop_count": 0.0, 
            
    
    """

    for place_name, priority, country_code in sorted(place_list, key=lambda x: x[0]):


        ################################################################
        ## Check if the city has already been processed
        ################################################################
        print(f"Processing: {place_name} ({priority}, {country_code})")
        output_file = f"{output_path}{place_name} PoI accessibility.json.gz"

        if os.path.isfile(output_file):
            print("Already processed")
            continue

        ################################################################
        ## Load population data from WorldPop
        ################################################################
        result = {}
        input_raster = f'{worldpop_path}{country_code.lower()}_ppp_2020_constrained.tif'
        if not os.path.isfile(input_raster):
            url = country_tif_urls[country_code]
            print(place_name, country_code)
            print(f'Downloading {place_name} population data from {url}...')

            # Perform the download
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raise an error if the download failed

                # Write the downloaded file to the output path
                with open(input_raster, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f'Successfully downloaded {input_raster}')

            except requests.exceptions.RequestException as e:
                print(f"Error downloading {place_name} data: {e}")

        worldpop = rioxarray.open_rasterio(input_raster)

        ################################################################
        ## Load checkpoint if available
        ################################################################
        processed_nodes = set([])
        checkpoint_file = f"{output_path}{place_name.split(',')[0]}_checkpoint.json"
        if CHECKPOINT:
            result = load_checkpoint(checkpoint_file)
            processed_nodes = set(result.keys())

        ################################################################
        ## Create the city graph
        ################################################################
        graphml_filename = f"{output_path}{place_name}.graphml"
        gzip_filename = f"{output_path}{place_name}.graphml.gz"

        if not os.path.isfile(gzip_filename):
            cityG = ox.graph_from_place(place_name, network_type="walk", retain_all=True)

            # Convert node IDs to string format
            node_mapping = {node: str(node) for node in cityG.nodes}
            cityG = nx.relabel_nodes(cityG, node_mapping)

            # Save and compress the city graph
            ox.save_graphml(cityG, graphml_filename)
            with open(graphml_filename, 'rb') as f_in, gzip.open(gzip_filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(graphml_filename)  # Remove uncompressed file

        else:
            cityG = ox.load_graphml(gzip_filename)
            node_mapping = {node: str(node) for node in cityG.nodes}
            cityG = nx.relabel_nodes(cityG, node_mapping)

        print("Number of nodes and edges:", cityG.number_of_nodes(), cityG.number_of_edges())

        ################################################################
        ## Create the extended city graph
        ################################################################
        extended_gzip_filename = f"{output_path}{place_name} extended.graphml.gz"
        extended_graphml_filename = f"{output_path}{place_name} extended.graphml"

        if not os.path.isfile(extended_gzip_filename):
            max_distance = int((MAX_TRAVEL_TIME * 60) * travel_speed / 3.6)
            bbox = get_extended_bebop_from_graph(cityG, max_distance)
            G = ox.graph_from_bbox(bbox=bbox, network_type="walk", retain_all=True)

            print(f"BBox and graph obtained: {bbox}")
            node_mapping = {node: str(node) for node in G.nodes}
            G = nx.relabel_nodes(G, node_mapping)

            for node, data in G.nodes(data=True):
                if node in cityG.nodes:
                    population_obj = worldpop.sel(x=data['x'], y=data['y'], method="nearest")
                    data['worldpop_count'] = float(max(population_obj.values[0], 0))
                    data['worldpop_code'] = f"{population_obj.coords['x'].values},{population_obj.coords['y'].values}"

            # Save and compress the extended graph
            ox.save_graphml(G, extended_graphml_filename)
            with open(extended_graphml_filename, 'rb') as f_in, gzip.open(extended_gzip_filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(extended_graphml_filename)  # Remove uncompressed file

        else:
            G = ox.load_graphml(extended_gzip_filename)
            node_mapping = {node: str(node) for node in G.nodes}
            G = nx.relabel_nodes(G, node_mapping)

            for node, data in G.nodes(data=True):
                if node in cityG.nodes:
                    data['worldpop_count'] = float(G.nodes[node]['worldpop_count'])
            for source, target, data in G.edges(data=True):
                data["length"] = float(data["length"])

        print("Extended number of nodes and edges:", G.number_of_nodes(), G.number_of_edges())

        nodes_not_in_g = [str(n) for n in cityG.nodes() if n not in G]
        print("Nodes not in extended BBOX:", nodes_not_in_g)

        ################################################################
        ## Retrieve Points of Interest (PoIs) within the extended graph
        ################################################################
        pois_filename = f"{output_path}{place_name}.csv.gz"

        if not os.path.isfile(pois_filename):
            bbox = get_extended_bebop_from_graph(G, int((MAX_TRAVEL_TIME * 60) * travel_speed / 3.6))
            pois = ox.features_from_bbox(bbox=bbox, tags=get_queryable_tags(categories))

            pois = pois.filter(['public_transport', 'highway', 'amenity', 'landuse', 'leisure', 'sport', 'geometry'])

            with gzip.open(pois_filename, 'wt', encoding='utf-8') as f:
                pois.to_csv(f, index=False)
        else:
            with gzip.open(pois_filename, 'rt', encoding='utf-8') as f:
                pois = pd.read_csv(f)
                pois['geometry'] = pois['geometry'].apply(wkt.loads)
                pois = gpd.GeoDataFrame(pois, geometry='geometry')

        pois_by_tags = {category: filter_pois_by_tags(pois, data['tags']) for category, data in categories.items()}
        print("Number of PoIs:", pois.count())

        ################################################################
        ## Start parallelized computation
        ################################################################
        print("Creating batches of nodes")
        nodes_to_process = [
            str(node) for node in cityG.nodes
            if str(node) not in processed_nodes  # Exclude already processed nodes
               and str(node) not in nodes_not_in_g  # Exclude missing OSM nodes
        ]

        batch_size = BATCH_SIZE
        nodes_batches = list(chunked(nodes_to_process, batch_size))

        print("Setting up parallelization")
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_batch = {
                executor.submit(compute_batch, G, pois_by_tags, batch, travel_speed, MAX_TRAVEL_TIME): batch
                for batch in nodes_batches
            }

            with tqdm(total=len(nodes_to_process)) as pbar:
                for future in concurrent.futures.as_completed(future_to_batch):
                    for node, proximity, density, entropy, coordinates in future.result():
                        if node in G:
                            result[node] = {
                                "coordinate": coordinates,
                                "proximity": proximity,
                                "density": density,
                                "entropy": entropy,
                                "worldpop_count": float(G.nodes[node]['worldpop_count']),
                                "worldpop_code": str(G.nodes[node]['worldpop_code'])
                            }

                    pbar.update(len(future_to_batch[future]))

                    # Periodic checkpoint saving
                    if len(result) % 1000000 == 0 and CHECKPOINT:
                        save_checkpoint_async(result, checkpoint_file)

        # Save final results
        with gzip.open(output_file, "wt") as f:
            json.dump(result, f, indent=4)

        print(len(result.keys()))
        print(f"Results saved to {output_file}")

        # Free memory after processing each city
        del worldpop, G, pois, pois_by_tags, nodes_batches
        gc.collect()
