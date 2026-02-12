#### Imports ####

import json

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import auc


#### Functions ####

def compute_min_90(threshold_range, cumulative_data):
    # Find the threshold corresponding to the first cumulative_data value >= 90
    for i, val in enumerate(cumulative_data):
        if val >= 90.0:
            return threshold_range[i]
    return None


def compute_auc(threshold_range, cumulative_data):
    # Compute AUC using trapezoidal rule
    return auc(threshold_range, cumulative_data)


def select_representative_cities(city_data):
    # Prepare data structures to store percentiles and AUC
    thresholds_90 = []
    aucs = []

    for city, data in city_data.items():
        threshold_range = np.array(data['threshold_range'])
        cumulative_data = np.array(data['cumulative_data'])

        # Compute the value corresponding to cumulative_data >= 90%
        threshold_90 = compute_min_90(threshold_range, cumulative_data)
        if threshold_90 is not None:
            thresholds_90.append(threshold_90)

        # Compute the AUC of the cumulative distribution
        auc_value = compute_auc(threshold_range, cumulative_data)
        aucs.append(auc_value)

    # Calculate percentiles
    auc_percentiles = np.percentile(aucs, np.arange(20, 91, 20))
    t90_percentiles = np.percentile(thresholds_90, np.arange(20, 91, 20))

    # Include the min and max
    auc_percentiles = np.concatenate(([min(aucs)], auc_percentiles, [max(aucs)]))
    print(f"N. of percentiles is {len(auc_percentiles)}")
    t90_percentiles = np.concatenate(([min(thresholds_90)], t90_percentiles, [max(thresholds_90)]))

    # Find the cities closest to these percentiles - AUC
    selected_cities_auc = {}
    for percentile in auc_percentiles:
        closest_idx = (np.abs(np.array(aucs) - percentile)).argmin()
        closest_city = list(city_data.keys())[closest_idx]
        selected_cities_auc[percentile] = closest_city

    # Find the cities closest to these percentiles - T90
    selected_cities_t90 = {}
    for percentile in t90_percentiles:
        closest_idx = (np.abs(np.array(thresholds_90) - percentile)).argmin()
        closest_city = list(city_data.keys())[closest_idx]
        selected_cities_t90[percentile] = closest_city

    return selected_cities_auc, selected_cities_t90, auc_percentiles, thresholds_90


def plot_cumulative_curves(city_data, selected_cities, img_path="./", data_path="./"):
    """
    This function plots the cumulative distribution curves for selected cities,
    orders the cities in the legend by their AUC values,
    and saves the plot and data in JSON format.
    """

    # Define markers, colors, and linestyles
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', '<', '>', 'h']
    linestyles = ['-']
    colors = plt.cm.tab20.colors

    to_save = {}
    auc_values = []

    # Compute AUC for selected cities
    city_auc = {}
    for city in selected_cities:
        threshold_range = np.array(city_data[city]['threshold_range'])
        cumulative_data = np.array(city_data[city]['cumulative_data'])
        city_auc[city] = auc(threshold_range, cumulative_data)

    # Sort cities by AUC value (low to high)
    sorted_cities = sorted(city_auc.keys(), key=lambda x: city_auc[x], reverse=True)

    plt.figure(figsize=(10, 6))

    # Plot each city's cumulative curve
    for i, city in enumerate(sorted_cities):
        threshold_range = np.array(city_data[city]['threshold_range'])
        cumulative_data = np.array(city_data[city]['cumulative_data'])

        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        alpha = 0.5  # Adjust transparency if needed
        label = city.split(',')[0]

        if 'Kommune' in label:
            label = label.split(' ')[0]

        plt.plot(threshold_range, cumulative_data, marker=marker, markersize=6, color=color, linestyle=linestyle,
                 label=label, alpha=alpha)

        # Save city data
        to_save[city] = {
            "threshold_range": list(map(float, threshold_range)),
            "cumulative_data": list(map(float, cumulative_data))
        }

    # Plot settings
    plt.xlabel("Proximity Threshold")
    plt.ylabel("Cumulative Population (%)")
    plt.title("Cumulative Percentage of Population Up to Each Proximity Threshold")
    plt.yticks(np.arange(0, 101, 10))

    xticks = [1, 5, 10, 15, 20, 30, 60]
    xticklabels = ['1', '5', '10', '15', '20', '30', '>=60']
    plt.xticks(ticks=xticks, labels=xticklabels)
    plt.xlim(-1, 61)
    plt.grid(True)

    # Legend positioning
    num_columns = min(len(sorted_cities), 6)
    plt.subplots_adjust(bottom=0.35)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=num_columns, frameon=False)

    # Save JSON and image
    if img_path:
        plt.savefig(img_path + 'Pareto_proximity_colors.jpg', dpi=500, transparent=True)
        with open(data_path + "pareto_proximity_selected_auc.json", "w") as f:
            json.dump(to_save, f, indent=4)

    plt.show()


# %%
#### Load Data ####

PATH = 'json/pareto_proximity.json'  # Add your path to data

with open(PATH, 'r') as in_file:
    data = json.load(in_file)

#### Preprocessing ####

# Remove last data point - This corresponds to a bigger category of minutes (60+)
for k in data.keys():
    data[k]['threshold_range'] = data[k]['threshold_range'][:-1]
    data[k]['cumulative_data'] = data[k]['cumulative_data'][:-1]

# Select the 81 cities used in the manuscript - Cities w/ closeness information
cities_to_keep = [
    "Addis Ababa, Ethiopia",
    "Adelaide, Australia",
    "Amsterdam, Noord-Holland, Netherlands",
    "Atlanta, Georgia, USA",
    "Auckland, New Zealand",
    "Bangkok, Thailand",
    "Barcelona, Spain",
    "Bari, Italy",
    "Berlin, Germany",
    "Bogota, Colombia",
    "Bologna, Italy",
    "Boston, Massachusetts,USA",
    "Brisbane, Australia",
    "Budapest, Hungary",
    "Buenos Aires, Argentina",
    "Calgary, Canada",
    "Catania, Italy",
    "Chicago, United States",
    "City of Los Angeles, United States",
    "City of New York City, United States",
    "City of Prague, Czechia",
    "Copenhagen Kommune, Denmark",
    "Dallas, Texas, USA",
    "Detroit, Michigan, USA",
    "Dublin, Ireland",
    "Edinburgh, United Kingdom",
    "Edmonton, Canada",
    "Florence, Italy",
    "Fortaleza, Brazil",
    "Fukuoka, Japan",
    "Genoa, Italy",
    "Greater London, United Kingdom",
    "Helsinki, Finland",
    "Houston, United States",
    "Istanbul, Turkey",
    "Jakarta, Indonesia",
    "Lisbon, Portugal",
    "Madrid, Spain",
    "Manchester, United Kingdom",
    "Manila, Philippines",
    "Melbourne, City of Melbourne, Victoria, Australia",
    "Mexico City, Mexico",
    "Miami, United States",
    "Milan, Italy",
    "Milwaukee, USA",
    "Minneapolis, Minnesota, USA",
    "Montreal (region administrative), Canada",
    "Munich, Germany",
    "Municipality of Athens, Greece",
    "Nairobi, Kenya",
    "Naples, Italy",
    "Nottingham, United Kingdom",
    "Osaka, Japan",
    "Oslo, Norway",
    "Ottawa, Canada",
    "Palermo, Italy",
    "Paris, France",
    "Philadelphia, United States",
    "Región Metropolitana de Santiago, Chile",
    "Rio de Janeiro, Brazil",
    "Rome, Italy",
    "Rotterdam, Netherlands",
    "San Antonio, Texas, USA",
    "San Diego, United States",
    "San Francisco, United States",
    "Sapporo, Japan",
    "Seattle, United States",
    "Singapore, Singapore",
    "Stockholm, Sweden",
    "Sydney, Australia",
    "Taipei, Taiwan",
    "Tallinn, Estonia",
    "The Hague, Netherlands",
    "Tokyo, Japan",
    "Toulouse, France",
    "Turin, Italy",
    "Vancouver, Canada",
    "Vienna, Austria",
    "Warsaw, Poland",
    "Washington, D.C., United States",
    "Zurich, Switzerland"
]

cities_to_remove = [c for c in data if c not in cities_to_keep]

for city in cities_to_remove:
    data.pop(city)

print("Total number of selected cities:", len(data))

# Compute auc curves and select representative cities
selected_cities_auc, selected_cities_t90, auc_percentiles, thresholds_90 = select_representative_cities(data)

print("Most representative cities - auc:", selected_cities_auc)
print("Most representative cities - t90:", selected_cities_t90)

# Plot cumulative curve of representative cities
plot_cumulative_curves(data, selected_cities_auc.values())