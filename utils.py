
from pyproj import Transformer, CRS
import pandas as pd
from shapely import Polygon

place_list = [
    ("Addis Ababa, Ethiopia",  1  , "ETH"),
    ("Adelaide, Australia ",  2  , "AUS"),
    ("Amsterdam, Noord-Holland, Netherlands ",  3  , "NLD"),
    ("Atlanta, Georgia, USA",  4  , "USA"),
    ("Auckland, New Zealand",  5  , "NZL"),
    ("Bangkok, Thailand",  6  , "THA"),
    ("Barcelona, Spain ",  7  , "ESP"),
    ("Bari, Italy",  8  , "ITA"),
    ("Beijing, China",  9  , "CHN"),
    ("Berlin, Germany",  10 , "DEU"),
    ("Bogota, Colombia ",  11 , "COL"),
    ("Bologna, Italy",  12 , "ITA"),
    ("Boston, Massachusetts,USA ",  13 , "USA"),
    ("Brisbane, Australia ",  14 , "AUS"),
    ("Budapest, Hungary",  15 , "HUN"),
    ("Buenos Aires, Argentina",  16 , "ARG"),
    ("Calgary, Canada",  17 , "CAN"),
    ("Catania, Italy",  18 , "ITA"),
    ("Chicago, United States ",  19 , "USA"),
    ("City of Cape Town, South Africa ",  20 , "ZAF"),
    ("City of Los Angeles, United States ",  21 , "USA"),
    ("City of New York City, United States",  22 , "USA"),
    ("City of Prague, Czechia",  23 , "CZE"),
    ("Copenhagen Kommune, Denmark",  24 , "DNK"),
    ("Dallas, Texas, USA",  25 , "USA"),
    ("Detroit, Michigan, USA ",  26 , "USA"),
    ("Dublin, Ireland",  27 , "IRL"),
    ("Edinburgh, United Kingdom ",  28 , "GBR"),
    ("Edmonton, Canada ",  29 , "CAN"),
    ("Florence, Italy",  30 , "ITA"),
    ("Fortaleza, Brazil",  31 , "BRA"),
    ("Fukuoka, Japan",  32 , "JPN"),
    ("Genoa, Italy",  33 , "ITA"),
    ("Greater London, United Kingdom",  34 , "GBR"),
    ("Hanoi, Vietnam",  35 , "VNM"),
    ("Helsinki, Finland",  36 , "FIN"),
    ("Ho Chi Minh City, Vietnam ",  37 , "VNM"),
    ("Houston, United States ",  38 , "USA"),
    ("Istanbul, Turkey ",  39 , "TUR"),
    ("Jakarta, Indonesia",  40 , "IDN"),
    ("Lima Metropolitana, Lima, Perù",  41 , "PER"),
    ("Lisbon, Portugal ",  42 , "PRT"),
    ("Madrid, Spain ",  43 , "ESP"),
    ("Manchester, United Kingdom",  44 , "GBR"),
    ("Manila, Philippines ",  45 , "PHL"),
    ("Medellin, Colombia",  46 , "COL"),
    ("Melbourne, City of Melbourne, Victoria, Australia ",  47 , "AUS"),
    ("Mexico City, Mexico ",  48 , "MEX"),
    ("Miami, United States",  49 , "USA"),
    ("Milan, Italy",  50 , "ITA"),
    ("Milwaukee, USA",  51 , "USA"),
    ("Minneapolis, Minnesota, USA",  52 , "USA"),
    ("Montreal (region administrative), Canada ",  53 , "CAN"),
    ("Moscow, Russia",  54 , "RUS"),
    ("Mumbai, India ",  55 , "IND"),
    ("Munich, Germany",  56 , "DEU"),
    ("Municipality of Athens, Greece",  57 , "GRC"),
    ("Nairobi, Kenya",  58 , "KEN"),
    ("Naples, Italy ",  59 , "ITA"),
    ("Nottingham, United Kingdom",  60 , "GBR"),
    ("Osaka, Japan",  61 , "JPN"),
    ("Oslo, Norway",  62 , "NOR"),
    ("Ottawa, Canada",  63 , "CAN"),
    ("Palermo, Italy",  64 , "ITA"),
    ("Paris, France ",  65 , "FRA"),
    ("Philadelphia, United States",  66 , "USA"),
    ("Región Metropolitana de Santiago, Chile",  67 , "CHL"),
    ("Rio de Janeiro, Brazil ",  68 , "BRA"),
    ("Rome, Italy",  69 , "ITA"),
    ("Rotterdam, Netherlands ",  70 , "NLD"),
    ("San Antonio, Texas, USA",  71 , "USA"),
    ("San Diego, United States",  72 , "USA"),
    ("San Francisco, United States ",  73 , "USA"),
    ("São Paulo, Brazil",  74 , "BRA"),
    ("Sapporo, Japan",  75 , "JPN"),
    ("Seattle, United States ",  76 , "USA"),
    ("Seoul, South Korea",  77 , "KOR"),
    ("Shanghai, China",  78 , "CHN"),
    ("Singapore, Singapore",  79 , "SGP"),
    ("Stockholm, Sweden",  80 , "SWE"),
    ("Sydney, Australia",  81 , "AUS"),
    ("Taipei, Taiwan",  82 , "TWN"),
    ("Tallinn, Estonia ",  83 , "EST"),
    ("The Hague, Netherlands ",  84 , "NLD"),
    ("Tokyo, Japan",  85 , "JPN"),
    ("Toulouse, France ",  86 , "FRA"),
    ("Turin, Italy",  87 , "ITA"),
    ("Vancouver, Canada",  88 , "CAN"),
    ("Vienna, Austria",  89 , "AUT"),
    ("Warsaw, Poland",  90 , "POL"),
    ("Washington, D.C., United States ",  91 , "USA"),
    ("Zurich, Switzerland ",  92 , "CHE"),
]

city_to_continent = {
    # Europe
    "Bari, Italy": "Europe",
    "Bologna, Italy": "Europe",
    "Catania, Italy": "Europe",
    "Genoa, Italy": "Europe",
    "Florence, Italy": "Europe",
    "Naples, Italy": "Europe",
    "Milan, Italy": "Europe",
    "Palermo, Italy": "Europe",
    "Rome, Italy": "Europe",
    "Turin, Italy": "Europe",
    "Municipality of Athens, Greece": "Europe",
    "Zurich, Switzerland": "Europe",
    "The Hague, Netherlands": "Europe",
    "Paris, France": "Europe",
    "Toulouse, France": "Europe",
    "Dublin, Ireland": "Europe",
    "Manchester, United Kingdom": "Europe",
    "Barcelona, Spain": "Europe",
    "Lisbon, Portugal": "Europe",
    "Nottingham, United Kingdom": "Europe",
    "Copenhagen Kommune, Denmark": "Europe",
    "Stockholm, Sweden": "Europe",
    "Helsinki, Finland": "Europe",
    "Amsterdam, Noord-Holland, Netherlands": "Europe",
    "Warsaw, Poland": "Europe",
    "City of Prague, Czechia": "Europe",
    "Oslo, Norway": "Europe",
    "Vienna, Austria": "Europe",
    "Greater London, United Kingdom": "Europe",
    "Madrid, Spain": "Europe",
    "Istanbul, Turkey": "Europe",
    "Munich, Germany": "Europe",
    "Edinburgh, United Kingdom": "Europe",
    "Berlin, Germany": "Europe",
    "Budapest, Hungary": "Europe",
    "Tallinn, Estonia": "Europe",
    "Rotterdam, Netherlands": "Europe",
    "Moscow, Russia": "Europe",


    # North America
    "Vancouver, Canada": "North America",
    "San Francisco, United States": "North America",
    "Miami, United States": "North America",
    "Washington, D.C., United States": "North America",
    "Seattle, United States": "North America",
    "Philadelphia, United States": "North America",
    "Montreal (region administrative), Canada": "North America",
    "San Diego, United States": "North America",
    "Calgary, Canada": "North America",
    "Chicago, United States": "North America",
    "City of New York City, United States": "North America",
    "Houston, United States": "North America",
    "City of Los Angeles, United States": "North America",
    "Ottawa, Canada": "North America",
    "Milwaukee, USA": "North America",
    "Boston, Massachusetts,USA": "North America",
    "Minneapolis, Minnesota, USA": "North America",
    "Detroit, Michigan, USA": "North America",
    "Dallas, Texas, USA": "North America",
    "San Antonio, Texas, USA": "North America",
    "Atlanta, Georgia, USA": "North America",
    "Edmonton, Canada": "North America",

    # Latin America and Caribbean
    "Bogota, Colombia": "Latin America and Caribbean",
    "Mexico City, Mexico": "Latin America and Caribbean",
    "Rio de Janeiro, Brazil": "Latin America and Caribbean",
    "Región Metropolitana de Santiago, Chile": "Latin America and Caribbean",
    "Buenos Aires, Argentina": "Latin America and Caribbean",
    "Medellin, Colombia": "Latin America and Caribbean",
    "São Paulo, Brazil": "Latin America and Caribbean",
    "Fortaleza, Brazil": "Latin America and Caribbean",
    "Lima Metropolitana, Lima, Perù": "Latin America and Caribbean",

    # Africa
    "Nairobi, Kenya": "Africa",
    "City of Cape Town, South Africa": "Africa",
    "Addis Ababa, Ethiopia": "Africa",

    # Oceania
    "Auckland, New Zealand": "Oceania",
    "Melbourne, City of Melbourne, Victoria, Australia": "Oceania",
    "Adelaide, Australia": "Oceania",
    "Brisbane, Australia": "Oceania",
    "Sydney, Australia": "Oceania",

    # Asia
    "Manila, Philippines": "Asia",
    "Seoul, South Korea": "Asia",
    "Jakarta, Indonesia": "Asia",
    "Singapore, Singapore": "Asia",
    "Bangkok, Thailand": "Asia",
    "Beijing, China": "Asia",
    "Shanghai, China": "Asia",
    "Ho Chi Minh City, Vietnam": "Asia",
    "Tokyo, Japan": "Asia",
    "Osaka, Japan": "Asia",
    "Sapporo, Japan": "Asia",
    "Fukuoka, Japan": "Asia",
    "Mumbai, India": "Asia",
    "Hanoi, Vietnam": "Asia",
    "Taipei, Taiwan":"Asia"
}


country_tif_urls = {
    "PHL": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/PHL/phl_ppp_2020_constrained.tif",  # Philippines
    "ITA": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/ITA/ita_ppp_2020_constrained.tif",  # Italy
    "FRA": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/FRA/fra_ppp_2020_constrained.tif",  # Italy
    "ESP": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/ESP/esp_ppp_2020_constrained.tif",  # Spain
    "CHE": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/CHE/che_ppp_2020_constrained.tif",  # Switzerland
    "GRC": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/GRC/grc_ppp_2020_constrained.tif",  # Greece
    "SWE": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/SWE/swe_ppp_2020_constrained.tif",  # Sweden
    "TWN": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/TWN/twn_ppp_2020_constrained.tif",  # Taiwan
    "KOR": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/KOR/kor_ppp_2020_constrained.tif",  # South Korea
    "ARG": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/ARG/arg_ppp_2020_constrained.tif",  # Argentina
    "PRT": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/PRT/prt_ppp_2020_constrained.tif",  # Portugal
    "DEU": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/DEU/deu_ppp_2020_constrained.tif",  # Germany
    "HUN": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/HUN/hun_ppp_2020_constrained.tif",  # Hungary
    "DNK": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/DNK/dnk_ppp_2020_constrained.tif",  # Denmark
    "NLD": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/NLD/nld_ppp_2020_constrained.tif",  # Netherlands
    "RUS": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/RUS/rus_ppp_2020_constrained.tif",  # Russia
    "CAN": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/CAN/can_ppp_2020_constrained.tif",  # Canada
    "GBR": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/GBR/gbr_ppp_2020_constrained.tif",  # United Kingdom
    "USA": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/USA/usa_ppp_2020_constrained.tif",  # United States
    "CHL": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/CHL/chl_ppp_2020_constrained.tif",  # Chile
    "SGP": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/SGP/sgp_ppp_2020_constrained.tif",  # Singapore
    "JPN": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/JPN/jpn_ppp_2020_constrained.tif",  # Japan
    "COL": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/COL/col_ppp_2020_constrained.tif",  # Colombia
    "MEX": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/MEX/mex_ppp_2020_constrained.tif",  # Mexico
    "POL": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/POL/pol_ppp_2020_constrained.tif",  # Poland
    "PER": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/PER/per_ppp_2020_constrained.tif",  # Peru
    "TUR": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/TUR/tur_ppp_2020_constrained.tif",  # Turkey
    "IDN": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/IDN/idn_ppp_2020_constrained.tif",  # Indonesia
    "CHN": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/CHN/chn_ppp_2020_constrained.tif",  # China
    "ZAF": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/maxar_v1/ZAF/zaf_ppp_2020_constrained.tif",  # South Africa
    "THA": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/THA/tha_ppp_2020_constrained.tif",  # Thailand
    "BRA": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/BRA/bra_ppp_2020_constrained.tif",  # Brazil
    "CZE": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/CZE/cze_ppp_2020_constrained.tif",  # Czechia
    "FIN": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/FIN/fin_ppp_2020_constrained.tif",  # Finland
    "AUT": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/AUT/aut_ppp_2020_constrained.tif",  # Austria
    "VNM": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/VNM/vnm_ppp_2020_constrained.tif",  # Vietnam
    "KEN": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/maxar_v1/KEN/ken_ppp_2020_constrained.tif",  # Kenya
    "NOR": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/NOR/nor_ppp_2020_constrained.tif",  # Norway
    "AUS": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/AUS/aus_ppp_2020_constrained.tif",  # Australia
    "NZL": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/NZL/nzl_ppp_2020_constrained.tif",  # New Zealand
    "IRL": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/IRL/irl_ppp_2020_constrained.tif",  # Ireland
    "EST": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/EST/est_ppp_2020_constrained.tif",  # Estonia
    "ETH": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/maxar_v1/ETH/eth_ppp_2020_constrained.tif",  # Ethiopia
    "IND": "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/IND/ind_ppp_2020_constrained.tif",  # India
}


categories = {
    'mobility': {
        'tags': {
            'public_transport': ['station', 'stop_position', 'platform', 'stop_area', 'stop_area_group'],
            'highway': ['bus_stop'],
            'amenity': ['bus_station']
        }
    },
    'active_living': {
        'tags': {
            'leisure': [
                'fitness_centre', 'sports_centre', 'park', 'pitch', 'playground',
                'swimming_pool', 'garden', 'golf_course', 'ice_rink', 'dog_park',
                'nature_reserve', 'marina', 'fitness_station'
            ],
            'landuse': ['recreation_ground', 'skatepark', 'skate_park'],
            'sport': ['skateboard'],
            'amenity': ['bicycle_parking']
        }
    },
    'entertainment': {
        'tags': {
            'amenity': ['pub', 'bar', 'theatre', 'cinema', 'nightclub', 'events_venue']
        }
    },
    'food': {
        'tags': {
            'amenity': ['restaurant', 'cafe', 'food_court', 'marketplace', 'community_centre']
        }
    },
    'community': {
        'tags': {
            'amenity': ['library', 'social_facility', 'social_centre', 'townhall']
        }
    },
    'education': {
        'tags': {
            'amenity': ['school', 'childcare', 'child_care', 'kindergarten', 'university', 'college']
        }
    },
    'health_and_wellbeing': {
        'tags': {
            'amenity': ['pharmacy', 'dentist', 'clinic', 'hospital', 'doctors']
        }
    }
}


def get_queryable_tags(categories):
    """
    Retrieve a dictionary of tags queryable in osmnx from the given categories.

    Parameters:
    categories (dict): Dictionary of categories with tags.

    Returns:
    dict: A dictionary where keys are tag types and values are lists of tags.
    """
    # Initialize a dictionary to collect all tags
    queryable_tags = {}

    for category, data in categories.items():
        for key, values in data['tags'].items():
            if key not in queryable_tags:
                queryable_tags[key] = list(set(values))  # Use set to remove duplicates
            else:
                queryable_tags[key] += list(set(values))
    return queryable_tags


def get_extended_bebop_from_graph(G, max_distance_meters):
    """
    Obtain an extended Bounding Box of Points (BeBOP) from a graph.

    Parameters:
    subG (networkx.Graph): The graph from which to compute the BeBOP.
    max_distance_meters (float): Maximum distance to extend the bounding box by, in meters.

    Returns:
    shapely.geometry.box: The extended bounding box.
    """
    # Extract node coordinates (longitude, latitude)
    node_coords = [(data['x'], data['y']) for _, data in G.nodes(data=True)]

    if len(node_coords) == 0:
        raise ValueError("The graph has no nodes.")

    # Create the initial bounding box
    min_x, min_y, max_x, max_y = (min(p[0] for p in node_coords),
                                  min(p[1] for p in node_coords),
                                  max(p[0] for p in node_coords),
                                  max(p[1] for p in node_coords))

    # Convert bounding box coordinates to projected coordinates (meters)
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    min_x_proj, min_y_proj = transformer.transform(min_x, min_y)
    max_x_proj, max_y_proj = transformer.transform(max_x, max_y)

    # Extend the bounding box in projected coordinates
    buffer_x = max_distance_meters
    buffer_y = max_distance_meters

    min_x_proj -= buffer_x
    max_x_proj += buffer_x
    min_y_proj -= buffer_y
    max_y_proj += buffer_y

    # Convert extended bounding box back to geographic coordinates
    min_x, min_y = transformer.transform(min_x_proj, min_y_proj, direction='INVERSE')
    max_x, max_y = transformer.transform(max_x_proj, max_y_proj, direction='INVERSE')

    # Return the extended bounding box
    extended_bbox = (max_y, min_y, max_x, min_x)

    return extended_bbox


def filter_pois_by_tags(pois, tags):
    """
    Filter POIs based on the provided tags, retaining rows that satisfy at least one filter.

    Parameters:
    - pois (pd.DataFrame): DataFrame containing POIs with various tags.
    - tags (dict): Dictionary where keys are tag types and values are lists of tag values to filter by.

    Returns:
    - pd.DataFrame: Filtered POIs that match at least one of the tag criteria.
    """
    # Convert tags dictionary to a set of conditions for filtering
    conditions = []
    for key, values in tags.items():
        if key in pois.columns:
            condition = pois[key].isin(values)
            conditions.append(condition)


    if not conditions:
        raise ValueError("No valid tags found in POIs DataFrame columns.")
    # Combine all conditions with logical OR
    combined_condition = pd.concat(conditions, axis=1).any(axis=1)

    # Apply combined condition to filter POIs
    filtered_pois = pois[combined_condition]

    return filtered_pois


def filter_pois_by_polygon(pois, polygon):
    """
    Filter POIs to include only those within the given polygon.
    """

    # Filter the POIs
    pois_within_polygon = pois[pois.geometry.intersects(polygon)]

    return pois_within_polygon


def calculate_area_in_square_km(polygon):
    """
    Calcola l'area di un dato poligono in chilometri quadrati (km²).

    Parametri:
    polygon (shapely.geometry.Polygon): Il poligono per il quale calcolare l'area.

    Restituisce:
    float: L'area del poligono in chilometri quadrati (km²).
    """
    if isinstance(polygon, Polygon):
        # Definisci la proiezione per convertire da WGS84 a Web Mercator
        wgs84 = CRS.from_epsg(4326)  # WGS84 latitudine/longitudine
        web_mercator = CRS.from_epsg(3857)  # Proiezione Web Mercator

        # Trasformatore per convertire da WGS84 a Web Mercator
        transformer = Transformer.from_crs(wgs84, web_mercator, always_xy=True)

        # Proietta il poligono in Web Mercator
        projected_coords = [transformer.transform(x, y) for x, y in polygon.exterior.coords]
        projected_polygon = Polygon(projected_coords)

        # Calcola l'area in metri quadrati
        area_m2 = projected_polygon.area

        # Converte l'area in km²
        area_km2 = area_m2 / 1000000
    else:
        area_km2 = None

    return area_km2

