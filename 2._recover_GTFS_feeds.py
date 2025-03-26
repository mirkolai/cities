import json
import os.path
import time
import osmnx as ox
import requests
from shapely import MultiPoint
import geopandas as gpd
import utils

# Define paths for storing GTFS data
gtfs_path = 'data/GTFS/'
os.makedirs(gtfs_path, exist_ok=True)

gtfs_feed_path = 'data/GTFS/Feeds/'
os.makedirs(gtfs_feed_path, exist_ok=True)

# Configure OSMnx to use cache
ox.config(use_cache=True)

# API Configuration
API_KEY = ""
BASE_URL = "https://api.transit.land"


def my_request(url, params):
    """
    Perform a GET request with a delay to avoid rate limiting.
    Prints the URL and JSON response.
    """
    time.sleep(2)  # Prevent excessive API calls
    print(url)
    response = requests.get(url, params=params)
    print(response.json())
    return response


def my_geocode_to_gdf(city_name):
    """
    Geocode a city name and return its corresponding GeoDataFrame.
    """
    city_gdf = ox.geocode_to_gdf(city_name)
    return city_gdf


def my_graph_from_place(city_name):
    """
    Retrieve a street network graph for a given city.
    """
    cityG = ox.graph_from_place(city_name)
    return cityG


def my_graph_to_gdfs(cityG):
    """
    Convert a street network graph into separate node and edge GeoDataFrames.
    """
    nodes, edges = ox.graph_to_gdfs(cityG)
    return nodes, edges


def get_operators_by_bbox(city_name):
    """
    Retrieve transit operators within a bounding box of a given city using pagination.
    Handles cases where the bounding box is too large by reducing its size iteratively.
    """
    url = f"{BASE_URL}/api/v2/rest/agencies?api_key={API_KEY}"
    all_operators = []  # List to store all operators
    city_gdf = my_geocode_to_gdf(city_name)

    # Extract bounding box from the city's GeoDataFrame
    minx, miny, maxx, maxy = city_gdf.total_bounds
    print(f"{miny},{minx},{maxy},{maxx}")
    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "per_page": 100,
    }
    i = 0

    while url:
        # Make the API request
        response = my_request(url, params=params)
        data = response.json()
        url = None  # Reset URL to break loop unless updated

        if "error" in data:
            print(data["error"])
            if data["error"] == "bbox too large":
                url = f"{BASE_URL}/api/v2/rest/agencies?api_key={API_KEY}"
                cityG = my_graph_from_place(city_name)
                nodes, edges = my_graph_to_gdfs(cityG)

                points = MultiPoint(list(zip(nodes.geometry.x, nodes.geometry.y)))
                s = gpd.GeoSeries([points], crs='EPSG:4326')
                hull = s.concave_hull(ratio=0.01255)
                city_boundary = gpd.GeoDataFrame(geometry=hull, crs=nodes.crs)

                # Adjust bounding box
                minx, miny, maxx, maxy = city_boundary.total_bounds
                width, height = maxx - minx, maxy - miny
                width_reduction, height_reduction = width * i / 2, height * i / 2
                minx += width_reduction
                maxx -= width_reduction
                miny += height_reduction
                maxy -= height_reduction
                i += 0.1
                params["bbox"] = f"{minx},{miny},{maxx},{maxy}"
                print(f"{miny},{minx},{maxy},{maxx}")

        # Collect operator data
        operators = data.get("agencies", [])
        all_operators += operators

        # Check for pagination
        if 'meta' in data and 'next' in data["meta"]:
            url = data["meta"]["next"]

    # Save collected operator data to JSON
    json_dict = {str(index + 1): item for index, item in enumerate(all_operators)}
    with open(f"{gtfs_feed_path}{city_name}.json", "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=4, ensure_ascii=False)

    print(f"Number of distinct onestop_id values: {len(operators)}")


def get_feed(feed_key, country, city_name):
    """
    Download the latest GTFS feed for a given transit operator.
    """
    url = f"{BASE_URL}/api/v2/rest/feeds/{feed_key}/download_latest_feed_version?api_key={API_KEY}"
    filename = f"{gtfs_path}{feed_key}-{country}-{city_name}.zip"

    if not os.path.isfile(filename):
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Save the file in chunks
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        time.sleep(30)  # Delay to prevent rate limiting


# Process cities from the utils.place_list
for city_name, size, country in sorted(utils.place_list, key=lambda x: x[1]):
    print(city_name)

    if os.path.isfile(f"{gtfs_feed_path}{city_name}.json"):
        print("already done")
        continue

    # Retrieve transit operators in the city's bounding box
    get_operators_by_bbox(city_name)
