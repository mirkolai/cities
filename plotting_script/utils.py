def round_to_significant(number):
    number = int(number)
    magnitude = 10 ** (len(str(number)) - 1)
    rounded_number = round(number / magnitude) * magnitude
    return rounded_number


def scale_value(value, domain_min, domain_max, range_min, range_max):

    scaled_value = range_min + (value - domain_min) * (range_max - range_min) / (domain_max - domain_min)

    scaled_value = max(range_min, min(range_max, scaled_value))
    return int(scaled_value)

istat_codes={
"Bari": "072006",
"Bologna": "037006",
"Catania": "087015",
"Genoa": "010025",
"Florence": "048017",
"Naples": "063049",
"Milan": "015146",
"Palermo": "082053",
"Rome": "058091",
"Turin": "001272"
}

cluster_colors =[
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
    "#ffff99",
    "#b15928"
]

city_to_continent = {

    "Bari": "Europe",
    "Bologna": "Europe",
    "Catania": "Europe",
    "Genoa": "Europe",
    "Florence": "Europe",
    "Naples": "Europe",
    "Milan": "Europe",
    "Palermo": "Europe",
    "Rome": "Europe",
    "Turin": "Europe",
    "Athens": "Europe",
    "Zurich": "Europe",
    "The Hague": "Europe",
    "Paris": "Europe",
    "Toulouse": "Europe",
    "Dublin": "Europe",
    "Manchester": "Europe",
    "Barcelona": "Europe",
    "Lisbon": "Europe",
    "Nottingham": "Europe",
    "Copenhagen": "Europe",
    "Stockholm": "Europe",
    "Helsinki": "Europe",
    "Amsterdam": "Europe",
    "Warsaw": "Europe",
    "Prague": "Europe",
    "Oslo": "Europe",
    "Vienna": "Europe",
    "London": "Europe",
    "Madrid": "Europe",
    "Istanbul": "Europe",
    "Munich": "Europe",
    "Edinburgh": "Europe",
    "Berlin": "Europe",
    "Budapest": "Europe",
    "Tallinn": "Europe",
    "Rotterdam": "Europe",
    "Moscow": "Europe",

    "Vancouver": "North America",
    "San Francisco": "North America",
    "Miami": "North America",
    "Washington": "North America",
    "Seattle": "North America",
    "Philadelphia": "North America",
    "Montreal": "North America",
    "San Diego": "North America",
    "Calgary": "North America",
    "Chicago": "North America",
    "New York City": "North America",
    "Houston": "North America",
    "Los Angeles": "North America",
    "Ottawa": "North America",
    "Milwaukee": "North America",
    "Boston": "North America",
    "Minneapolis": "North America",
    "Detroit": "North America",
    "Dallas": "North America",
    "San Antonio": "North America",
    "Atlanta": "North America",
    "Edmonton": "North America",

    "Bogota": "Latin America and Caribbean",
    "Mexico City": "Latin America and Caribbean",
    "Rio de Janeiro": "Latin America and Caribbean",
    "Santiago": "Latin America and Caribbean",
    "Buenos Aires": "Latin America and Caribbean",
    "Medellin": "Latin America and Caribbean",
    "São Paulo": "Latin America and Caribbean",
    "Fortaleza": "Latin America and Caribbean",
    "Lima": "Latin America and Caribbean",

    "Nairobi": "Africa",
    "Cape Town": "Africa",
    "Addis Ababa": "Africa",

    "Auckland": "Oceania",
    "Melbourne": "Oceania",
    "Adelaide": "Oceania",
    "Brisbane": "Oceania",
    "Sydney": "Oceania",

    "Manila": "Asia",
    "Seoul": "Asia",
    "Jakarta": "Asia",
    "Singapore": "Asia",
    "Bangkok": "Asia",
    "Beijing": "Asia",
    "Shanghai": "Asia",
    "Ho Chi Minh City": "Asia",
    "Tokyo": "Asia",
    "Osaka": "Asia",
    "Sapporo": "Asia",
    "Fukuoka": "Asia",
    "Mumbai": "Asia",
    "Hanoi": "Asia",
    "Taipei":"Asia"
}

continent_to_color={

    "North America": "#FFD700",#
    "Latin America and Caribbean":	"#ff7f00",#
    "Europe":	"#984ea3",#
    "Africa":	"#4daf4a",#
    "Asia":	"#377eb8"	,#
    "Oceania":	"#e41a1c",#
}
