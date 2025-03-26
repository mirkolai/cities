import json

import pandas as pd
import osmnx as ox
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os
import utils

"""
In this script, starting from the income data provided by the Italian Ministry of Economy and Finance, we associate an average income with each node. 
It is important to highlight that the data is aggregated by postal code (CAP), but the boundaries of Italian postal codes are not available for free. Therefore, the script attempts to infer the postal code of an intersection by checking the most frequent postal code among the nearby addresses retrievable from OSM.

The script produces a file for each city, {place_name} income.json, which contains a dictionary that associates the intersection ID (key) with the average income (value).
"""

def weighted_variance(df, mean_col):
    # Columns for income and frequencies
    """income_columns = [
        'Total income less than or equal to zero euros - Amount in euros',
        'Total income from 0 to 10,000 euros - Amount in euros',
        'Total income from 10,000 to 15,000 euros - Amount in euros',
        'Total income from 15,000 to 26,000 euros - Amount in euros',
        'Total income from 26,000 to 55,000 euros - Amount in euros',
        'Total income from 55,000 to 75,000 euros - Amount in euros',
        'Total income from 75,000 to 120,000 euros - Amount in euros',
        'Total income over 120,000 euros - Amount in euros'
    ]"""
    income_columns = [
        'Reddito complessivo minore o uguale a zero euro - Ammontare in euro',
        'Reddito complessivo da 0 a 10000 euro - Ammontare in euro',
        'Reddito complessivo da 10000 a 15000 euro - Ammontare in euro',
        'Reddito complessivo da 15000 a 26000 euro - Ammontare in euro',
        'Reddito complessivo da 26000 a 55000 euro - Ammontare in euro',
        'Reddito complessivo da 55000 a 75000 euro - Ammontare in euro',
        'Reddito complessivo da 75000 a 120000 euro - Ammontare in euro',
        'Reddito complessivo oltre 120000 euro - Ammontare in euro'
    ]
    """
    frequency_columns = [
    'Total income less than or equal to zero euros - Frequency',
    'Total income from 0 to 10,000 euros - Frequency',
    'Total income from 10,000 to 15,000 euros - Frequency',
    'Total income from 15,000 to 26,000 euros - Frequency',
    'Total income from 26,000 to 55,000 euros - Frequency',
    'Total income from 55,000 to 75,000 euros - Frequency',
    'Total income from 75,000 to 120,000 euros - Frequency',
    'Total income over 120,000 euros - Frequency'
    ]
    """
    frequency_columns = [
        'Reddito complessivo minore o uguale a zero euro - Frequenza',
        'Reddito complessivo da 0 a 10000 euro - Frequenza',
        'Reddito complessivo da 10000 a 15000 euro - Frequenza',
        'Reddito complessivo da 15000 a 26000 euro - Frequenza',
        'Reddito complessivo da 26000 a 55000 euro - Frequenza',
        'Reddito complessivo da 55000 a 75000 euro - Frequenza',
        'Reddito complessivo da 75000 a 120000 euro - Frequenza',
        'Reddito complessivo oltre 120000 euro - Frequenza'
    ]

    # Differenza tra i redditi e la media calcolata
    squared_diffs = [(df[income_col] - df[mean_col]) ** 2 for income_col in income_columns]

    # Moltiplica la differenza al quadrato per la frequenza
    weighted_squared_diffs = [diff * df[frequency_col] for diff, frequency_col in zip(squared_diffs, frequency_columns)]

    # Somma le varianze ponderate e le frequenze
    sum_weighted_squared_diffs = sum(weighted_squared_diffs)
    sum_frequencies = df[frequency_columns].sum(axis=1)

    # Calcola la varianza ponderata
    variance = sum_weighted_squared_diffs / sum_frequencies

    return variance


def get_n_nearest_caps(sindex, gdf_points, point, n):
    # Get the bounding box of the point with a small buffer to limit the search area
    #If your geometries are in a projected CRS (like UTM), the units are in meters. A buffer of 100 would correspond to 100 meters.
    buffer_distance = 500 # Adjust as necessary
    bbox = point.buffer(buffer_distance).envelope
    #print(bbox)
    # Use the spatial index to find candidate points within the bounding box
    candidate_indices = list(sindex.query(bbox))
    # Create a GeoDataFrame with the candidate points
    candidate_points = gdf_points.iloc[candidate_indices]

    # Calculate distances from the point to the candidate points
    distances = candidate_points.geometry.distance(point)

    # Get the indices of the N nearest points
    nearest_indices = distances.nsmallest(n).index
    nearest_points = candidate_points.loc[nearest_indices]

    # Count occurrences of each CAP
    cap_counts = nearest_points['addr:postcode'].value_counts()

    # Determine the most frequent CAP
    if len(cap_counts) > 1:  # If there are different CAPs
        most_frequent_cap = cap_counts.idxmax()
        frequency = cap_counts.max()
        return most_frequent_cap, frequency
    elif len(cap_counts) ==1:  # If there's only one CAP
        return nearest_points['addr:postcode'].iloc[0], 1
    else:  # If there's only one CAP
        return None, None


# ​To load the CSV file containing income data, which is available via a link provided in the README.txt
df_redditi = pd.read_csv('data/income/Redditi_e_principali_variabili_IRPEF_su_base_subcomunale_CSV_2022.csv',
                         dtype={"CAP":str}, sep=';')

df_redditi.fillna(0, inplace=True)


# 'Reddito complessivo minore o uguale a zero euro - Frequenza', 'Reddito complessivo minore o uguale a zero euro - Ammontare in euro',
# 'Reddito complessivo da 0 a 10000 euro - Frequenza', 'Reddito complessivo da 0 a 10000 euro - Ammontare in euro',
# 'Reddito complessivo da 10000 a 15000 euro - Frequenza', 'Reddito complessivo da 10000 a 15000 euro - Ammontare in euro',
# 'Reddito complessivo da 15000 a 26000 euro - Frequenza', 'Reddito complessivo da 15000 a 26000 euro - Ammontare in euro',
# 'Reddito complessivo da 26000 a 55000 euro - Frequenza', 'Reddito complessivo da 26000 a 55000 euro - Ammontare in euro',
# 'Reddito complessivo da 55000 a 75000 euro - Frequenza', 'Reddito complessivo da 55000 a 75000 euro - Ammontare in euro',
# 'Reddito complessivo da 75000 a 120000 euro - Frequenza', 'Reddito complessivo da 75000 a 120000 euro - Ammontare in euro',
# 'Reddito complessivo oltre 120000 euro - Frequenza', 'Reddito complessivo oltre 120000 euro - Ammontare in euro',
df_redditi['Reddito medio'] = \
    (
            df_redditi['Reddito complessivo minore o uguale a zero euro - Ammontare in euro'] + df_redditi[
        'Reddito complessivo da 0 a 10000 euro - Ammontare in euro'] +
            df_redditi['Reddito complessivo da 10000 a 15000 euro - Ammontare in euro'] + df_redditi[
                'Reddito complessivo da 15000 a 26000 euro - Ammontare in euro'] +
            df_redditi['Reddito complessivo da 26000 a 55000 euro - Ammontare in euro'] + df_redditi[
                'Reddito complessivo da 55000 a 75000 euro - Ammontare in euro'] +
            df_redditi['Reddito complessivo da 75000 a 120000 euro - Ammontare in euro'] + df_redditi[
                'Reddito complessivo oltre 120000 euro - Ammontare in euro']
    )/(
            df_redditi['Reddito complessivo minore o uguale a zero euro - Frequenza']+df_redditi['Reddito complessivo da 0 a 10000 euro - Frequenza']+
            df_redditi['Reddito complessivo da 10000 a 15000 euro - Frequenza']+df_redditi['Reddito complessivo da 15000 a 26000 euro - Frequenza']+
            df_redditi['Reddito complessivo da 26000 a 55000 euro - Frequenza']+df_redditi['Reddito complessivo da 55000 a 75000 euro - Frequenza']+
            df_redditi['Reddito complessivo da 75000 a 120000 euro - Frequenza']+df_redditi['Reddito complessivo oltre 120000 euro - Frequenza']
     )


df_redditi['Varianza'] = weighted_variance(df_redditi, 'Reddito medio')

df_redditi['Deviazione standard'] = np.sqrt(df_redditi['Varianza'])

# Display the final result with standard deviation per postal code (CAP)
reddito_medio_per_cap = df_redditi.set_index('CAP')[['Reddito medio']].to_dict(orient='index')


for place_name,_,_ in utils.place_list:
    if "Italy" not in place_name:
        continue
    print(place_name)

    if os.path.isfile(f"income/{place_name} income new.json"):
        continue
    nodes_cap={}


    # Extract the geometry (buildings, roads, PoIs) that have the associated postal code (CAP)
    tags = {'addr:postcode': True}
    gdf = ox.features_from_place(place_name, tags)

    gdf_points = gdf[gdf.geometry.type == 'Point'].copy()


    #graph = ox.graph_from_place(place_name, network_type='all')
    path = f"output/{place_name}.graphml.gz"
    graph = ox.load_graphml(path)
    nodes, edges = ox.graph_to_gdfs(graph)

    # Transform the CRS to a projected one (EPSG:32632 is a good CRS for Italy)
    gdf_points = gdf_points.set_crs(epsg=4326)  # WGS 84
    gdf_points = gdf_points.to_crs(epsg=32632)  # to UTM zone 32N
    nodes = nodes.set_crs(epsg=4326)  # WGS 84
    nodes = nodes.to_crs(epsg=32632)  # to UTM zone 32N

    # Associate the average income to the nodes based on the postal code (CAP)
    nodes['CAP'] = None
    nodes['Reddito medio'] = None

    # Iterate through the nodes and associate the average income
    sindex = gdf_points.sindex

    for idx, row in nodes.iterrows():
        nearest_cap,frequency=get_n_nearest_caps(sindex,gdf_points, row.geometry,15)
        if nearest_cap not in reddito_medio_per_cap:
            nodes_cap[idx] = {"cap": None,
                              "income": None,
                              }
        elif nearest_cap is None:
            nodes_cap[idx] = {"cap": None,
                              "income": None,
                              }
        else:
            nodes_cap[idx]={ "cap" : nearest_cap,
                         "income": None if nearest_cap is None else reddito_medio_per_cap[str(nearest_cap)],
                         }

    file= open(f"income/{place_name} income.json","w")
    json.dump(nodes_cap,file,indent=4)
    file.close()


