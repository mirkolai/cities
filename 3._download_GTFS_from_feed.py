import json
import os.path
import time
import requests
import utils
from unidecode import unidecode

# API Configuration
API_KEY = ""
BASE_URL = "https://api.transit.land"

# Define paths for storing GTFS data
gtfs_path = 'data/GTFS/'
os.makedirs(gtfs_path, exist_ok=True)

gtfs_feed_path = 'data/GTFS/Feeds/'
os.makedirs(gtfs_feed_path, exist_ok=True)


def get_feed(feed_key, country, city_name):
    """
    Download the latest GTFS feed for a given transit operator.
    If the feed file does not already exist, it is downloaded and saved.
    """
    url = f"{BASE_URL}/api/v2/rest/feeds/{feed_key}/download_latest_feed_version?api_key={API_KEY}"
    filename = f"{gtfs_path}{feed_key}-{country}-{city_name}.zip"

    if not os.path.isfile(filename):
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        print(city_name, "starting")

        # Save the downloaded file in chunks
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        time.sleep(60)  # Delay to avoid API rate limits
    print(city_name, "done")


# Iterate through the list of cities and fetch GTFS feeds
for city_name, size, country in sorted(utils.place_list, key=lambda x: x[0]):
    filename = f"{gtfs_feed_path}{city_name}.json"
    print(city_name)
    print(filename)

    # Handle special case for Copenhagen
    if "Copenhagen" in city_name:
        city_name = city_name.replace("Copenhagen", "Kobenhavn")

    if os.path.isfile(filename):
        print(filename)
        feeds = set()
        df = json.load(open(filename))

        # First pass: Match by city name
        for key, item in df.items():
            go = 0
            if item["places"] is None:
                continue
            for place in item["places"]:
                if place["city_name"] is not None and unidecode(place["city_name"]).lower() in city_name.lower():
                    go += 1
            if go == 1:
                feeds.add(item["feed_version"]["feed"]["onestop_id"])

        # Second pass: Match by administrative level 1 (state/province)
        if len(feeds) == 0:
            for key, item in df.items():
                go = 0
                if item["places"] is None:
                    continue
                for place in item["places"]:
                    if place["adm1_name"] is not None and unidecode(place["adm1_name"]).lower() in city_name.lower():
                        go += 1
                if go == 1:
                    feeds.add(item["feed_version"]["feed"]["onestop_id"])

        # Third pass: Match by administrative level 0 (country)
        if len(feeds) == 0:
            for key, item in df.items():
                go = 0
                if item["places"] is None:
                    continue
                for place in item["places"]:
                    if place["adm0_name"] is not None and unidecode(place["adm0_name"]).lower() in city_name.lower():
                        go += 1
                if go == 1:
                    feeds.add(item["feed_version"]["feed"]["onestop_id"])

        print(city_name, len(feeds), feeds)

        if len(feeds) == 0:
            print("NO FEED", city_name)
            continue

        # Download GTFS feeds for identified operators
        for feed in feeds:
            print(feed)
            get_feed(feed, country, city_name)
    else:
        print(city_name, "missing")
