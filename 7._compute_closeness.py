import json
import os.path
import gzip
import networkx as nx
from tqdm import tqdm
from utils import place_list
import igraph as ig
import time

# Define the output directory and ensure it exists
output_path = 'output/'
os.makedirs(output_path, exist_ok=True)

# Iterate through the list of places, sorted by name
for place_name, _, country_code in sorted(place_list, key=lambda x: x[1]):
    weight = "weight"  # Attribute representing travel time

    start_time = time.time()  # Measure execution time

    closeness_path = f"{output_path}{place_name} closeness.json.gz"

    # Check if closeness results already exist to avoid recomputation
    if os.path.isfile(closeness_path):
        continue  # Skip already processed locations

    # Path to the road and transit network file
    graphml_path = f"{output_path}{place_name} - walk & transit.graphml.gz"

    # Ensure the necessary network file exists
    if not os.path.isfile(graphml_path):
        print(f"Missing transit network for {place_name}. Skipping...")
        continue


    # Open and load the network graph from a compressed GraphML file
    with gzip.open(graphml_path, 'rt', encoding='utf-8') as f:
        cityG = nx.read_graphml(f)

    ###### Preprocessing for iGraph ##########

    print(f"Ensuring '{weight}' attributes are numeric...")
    non_numeric_weights_count = 0

    # Validate and convert edge weights to numeric values
    for u, v, key, data in tqdm(cityG.edges(keys=True, data=True), desc="Processing edges"):
        try:
            data[weight] = float(data[weight])
            if data[weight] <= 0.0:
                non_numeric_weights_count += 1
        except ValueError:
            non_numeric_weights_count += 1
            print(f"Non-numeric weight found on edge ({u}, {v}, {key}): {data[weight]}")

    print(f"Total non-numeric weight values found: {non_numeric_weights_count}")

    # Remove edges with invalid weight values
    if non_numeric_weights_count > 0:
        print(f"Removing {non_numeric_weights_count} edges with invalid '{weight}' values.")
        edges_to_remove = [
            (u, v, key)
            for u, v, key, data in cityG.edges(keys=True, data=True)
            if (weight in data and not isinstance(data[weight], (int, float))) or
               (weight in data and data[weight] <= 0.0)
        ]
        for u, v, key in edges_to_remove:
            cityG.remove_edge(u, v, key)

    # Convert MultiDiGraph to DiGraph, keeping the shortest travel time
    print("Converting MultiDiGraph to DiGraph...")
    simple_cityG = nx.DiGraph()
    for u, v, data in cityG.edges(data=True):
        if not simple_cityG.has_edge(u, v):
            simple_cityG.add_edge(u, v, **data)
        else:
            # Keep the edge with the lowest weight (fastest travel time)
            if data[weight] < simple_cityG[u][v][weight]:
                simple_cityG.remove_edge(u, v)
                simple_cityG.add_edge(u, v, **data)
    print("Conversion to DiGraph completed.")

    #########################

    # Convert NetworkX graph to iGraph for efficient centrality computation
    edges = simple_cityG.edges(data=True)
    g = ig.Graph.TupleList(
        [(u, v, d[weight]) for u, v, d in edges],
        directed=True,
        edge_attrs=[weight]
    )

    # Compute closeness centrality (harmonic closeness)
    closeness = g.harmonic_centrality(weights=g.es[weight], normalized=True, mode="out")

    # Map results back to original node names
    node_names = g.vs['name']
    closeness_dict = {node_names[i]: {"closeness": closeness[i]} for i in range(len(node_names))}

    end_time = time.time()
    duration = end_time - start_time
    print(f"Processing complete. Time taken: {duration:.2f} seconds")

    # Save the computed closeness centrality in a compressed JSON file
    json.dump(closeness_dict, gzip.open(closeness_path, "wt"), indent=4)
