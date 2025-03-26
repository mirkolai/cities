import os.path
import gzip
import shutil
import networkx as nx
import osmnx as ox
from real_neighborhoods import RealNeighborhood
import pickle
from utils import place_list
import json

# Define the output directory and ensure it exists
output_path = 'output/'
os.makedirs(output_path, exist_ok=True)

# Iterate through the list of places, sorted by a specific attribute
for place_name, _, country_code in sorted(place_list, key=lambda x: x[1]):

    print(f"Processing: {place_name}")

    # Check if the Infomap clustering results already exist
    infomap_pickle_path = f"{output_path}{place_name} infomap.pickle"
    if os.path.isfile(infomap_pickle_path):
        print("Infomap results already exist. Skipping...")
        continue

    # Check if the road network file exists
    graphml_gz_path = f"{output_path}{place_name}.graphml.gz"
    if not os.path.isfile(graphml_gz_path):
        print("Missing road network file. Skipping...")
        continue

    # Temporary path for the decompressed GraphML file
    temp_graphml_path = f"{output_path}{place_name}_decompressed.graphml"

    # Decompress the GZIP-compressed GraphML file
    with gzip.open(graphml_gz_path, 'rt', encoding='utf-8') as f_in:
        with open(temp_graphml_path, 'w', encoding='utf-8') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Load the road network graph from the decompressed GraphML file
    G = ox.load_graphml(temp_graphml_path)

    # Remove the temporary file after loading the graph
    os.remove(temp_graphml_path)

    # Compute neighborhoods and node-module associations using Infomap clustering
    neighborhoods, node_module = RealNeighborhood().compute_real_neighborhood(G)

    # Save node-module mappings as a compressed JSON file
    infomap_json_path = f"{output_path}{place_name} infomap.json.gz"
    json.dump(node_module, gzip.open(infomap_json_path, "wt"), indent=4)

    # Save neighborhood data as a pickle file for further analysis
    pickle.dump(neighborhoods, open(infomap_pickle_path, "wb"))
