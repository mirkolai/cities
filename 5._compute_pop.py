import os.path
import numpy as np
from collections import defaultdict
import gzip
import json
from utils import place_list

# Define paths for required directories and create them if they do not exist
worldpop_path = 'data/worldpop/'
os.makedirs(worldpop_path, exist_ok=True)
output_path = 'output/'
os.makedirs(output_path, exist_ok=True)
gtfs_path = 'data/GTFS/'
os.makedirs(gtfs_path, exist_ok=True)

# Iterate over places sorted by a specific attribute
for place_name, _, country_code in sorted(place_list, key=lambda x: x[1]):
    print(place_name)

    Pop = {}
    pop_path = f"{output_path}{place_name} Pop.json.gz"

    # Check if the population data file already exists
    if os.path.isfile(pop_path):
        print("Already done")
        continue

    accessibility_path = f"{output_path}{place_name} PoI accessibility.json.gz"

    # Ensure that the accessibility file exists before proceeding
    if not os.path.isfile(accessibility_path):
        print("Missing accessibility")
        continue

    # Load accessibility data from a compressed JSON file
    PoIAccessibility = json.load(gzip.open(accessibility_path, 'rt', encoding='utf-8'))

    # Group points by population coordinates to aggregate population data
    grouped_population = defaultdict(list)
    for key, value in PoIAccessibility.items():
        pop_coords = value["worldpop_code"]
        grouped_population[pop_coords].append(key)

    # Calculate weighted proximity based on population for each point
    for pop_coords, points in grouped_population.items():
        # Compute total population in the group
        total_population_group = np.average([PoIAccessibility[point]["worldpop_count"] for point in points])
        population_per_point = total_population_group / len(points)

        # Assign the computed population to each point
        for point in points:
            Pop[point] = population_per_point

    # Save the computed population data as a compressed JSON file
    json.dump(Pop, gzip.open(pop_path, 'wt'), indent=4)
