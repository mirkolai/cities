import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

# Load the JSON data
data = json.load(open("json/income italian cities.json"))

# Extract data for plotting
cities = list(data.keys())

# select only italian cities
italian_cities = [city for city in cities if city.split(",")[-1].strip() == "Italy"]
# cities_to_exclude = ["Catania, Italy"]
cities_to_exclude = []

# exclude cities
for city in cities_to_exclude:
    italian_cities.remove(city)

italian_cities_indices = np.array([i for i, city in enumerate(cities) if city in italian_cities])   
# split the city name to get the city name only, and keep the country name in a separate list
city_names = np.array([city.split(",")[0].strip() for city in italian_cities])

city_to_area = {
    "Bari": "Southern Italy",
    "Milan": "Northern Italy",
    "Naples": "Southern Italy",
    "Palermo": "Southern Italy",
    "Rome": "Central Italy",
    "Turin": "Northern Italy",
    "Catania": "Southern Italy",
    "Florence": "Central Italy",
    "Genoa": "Northern Italy",
    "Bologna": "Northern Italy"
}


# assign to every city a region based on the areas
areas = [city_to_area[city] for city in city_names]

area_to_color={
"Southern Italy": "#67a9cf",#
"Central Italy": "#f7f7f7",#
"Northern Italy": "#ef8a62"#
}

# assign to every city a color based on the region
colors = np.array([area_to_color[area] for area in areas])

# Extract data for Italian cities
connectivity = np.array([data[city]["connectivity_norm"] for city in italian_cities])
accessibility = np.array([data[city]["Accessibility_norm"] for city in italian_cities])
income = np.array([data[city]["income"] for city in italian_cities])
# Normalize marker size
# sizes = income / max(income)*2000   # Scale for visibility
scaler = MinMaxScaler(feature_range=(100, 1000))
sizes = scaler.fit_transform(income.reshape(-1, 1)).flatten()



# calculate pearson correlation coefficient between connectivity and accessibility
pearson_corr, pearson_pvalue = pearsonr(connectivity, accessibility)
print("Pearson correlation coefficient between connectivity and accessibility:", pearson_corr)
print("Pearson correlation p-value between connectivity and accessibility:", pearson_pvalue)

# calculate kendall tau correlation coefficient between connectivity and accessibility
from scipy.stats import kendalltau
kendall_tau,  kendall_tau_pvalue = kendalltau(connectivity, accessibility)
print("Kendall tau correlation coefficient between connectivity and accessibility:", kendall_tau)
print("Kendall tau correlation p-value between connectivity and accessibility:", kendall_tau_pvalue)


# calculate mean and standard deviation for each city
mean_connectivity = np.mean(connectivity)
std_connectivity = np.std(connectivity)
mean_accessibility = np.mean(accessibility)
std_accessibility = np.std(accessibility)

# calculate median for each city
median_connectivity = np.median(connectivity)
median_accessibility = np.median(accessibility)

# calculate mean income and standard deviation for each city
mean_income = np.mean(income)
std_income = np.std(income)

# print mean and standard deviation for each city
print("Mean connectivity:", mean_connectivity)
print("Standard deviation connectivity:", std_connectivity)
print("Mean accessibility:", mean_accessibility)
print("Standard deviation accessibility:", std_accessibility)
print("Median connectivity:", median_connectivity)
print("Median accessibility:", median_accessibility)
print("Mean income:", mean_income)
print("Standard deviation income:", std_income)

import matplotlib.pyplot as plt
import numpy as np

# Create bubble chart
plt.figure(figsize=(8, 8))

# Plot all points
# Ordina per dimensione per disegnare prima i punti più grandi
sorted_indices = np.argsort(sizes)[::-1]
connectivity_sorted = connectivity[sorted_indices]
accessibility_sorted = accessibility[sorted_indices]
sizes_sorted = sizes[sorted_indices]
colors_sorted = colors[sorted_indices]
city_names_sorted = [city_names[i] for i in sorted_indices]

plt.scatter(connectivity_sorted, accessibility_sorted, s=sizes_sorted, color=colors_sorted, edgecolors='k')

# Annotare i punti con frecce e sfondo per le etichette
for i in range(len(connectivity_sorted)):
    xshift = 0.01
    yshift = 0.01
    if city_names_sorted[i] == "Bologna":
        xshift = 0
        yshift = -0.02
    if city_names_sorted[i] == "Turin":
        xshift = 0.02
        yshift = -0.01
    '''
    if city_names_sorted[i] == "Milan":
        xshift = -0.01
        yshift = 0.02
    if city_names_sorted[i] == "Bologna":
        xshift = -0.02
        yshift = -0.02
    if city_names_sorted[i] == "Turin":
        xshift = 0
        yshift = 0.02
    '''
    plt.annotate(city_names_sorted[i], 
                 (connectivity_sorted[i], accessibility_sorted[i]), 
                 xytext=(connectivity_sorted[i] + xshift, accessibility_sorted[i] + yshift),  # Offset
                 textcoords='data',
                 fontsize=9, 
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'), 
                 arrowprops=dict(arrowstyle="->", color='black', lw=0.8))

# Labels and Titles
plt.xlabel("Normalized closeness")
plt.gca().yaxis.set_label_position("right")
plt.gca().yaxis.tick_right()
plt.ylabel("PoI-accessibility")

# setting x and y axis limits
plt.xlim(min(connectivity)-min(connectivity)*0.1, max(connectivity)+max(connectivity)*0.1)
plt.ylim(min(accessibility)-min(accessibility)*0.1, max(accessibility)+max(accessibility)*0.1)

'''
# plot a different legend for the colors in the upper left corner
handles, labels = plt.gca().get_legend_handles_labels()
for i, area in enumerate(["Northern Italy", "Central Italy", "Southern Italy"]):  
    handles.append(plt.Line2D([], [], marker='o', color='w', markerfacecolor=area_to_color[area], markeredgecolor='k', markersize=10, label=area))
legend2 = plt.legend(handles=handles[len(handles)-len(colors):], scatterpoints=1, labelspacing=0.5, loc='upper left', handletextpad=1)
plt.gca().add_artist(legend2)  # Keep the first legend
# add a smooth grid
plt.grid(True, linestyle='--', alpha=0.5)
'''
# Plot population legend in lower right corner
scaler = MinMaxScaler(feature_range=(100, 1000))
sizes = scaler.fit_transform(np.array([min(income), np.mean(income), max(income)]).reshape(-1, 1)).flatten()
incsizes = zip([min(income), np.mean(income), max(income)], sizes)
for income, size in incsizes:
    plt.scatter([], [], edgecolors='k', facecolors='none', s=size, label=f'{income/1000:.2f} k€')


legend1 = plt.legend(title="Income", scatterpoints=1, labelspacing=0.5, loc='upper right', handletextpad=1.5, frameon=False)
plt.gca().add_artist(legend1)  # Keep the first legend


# remove right and top spines
#plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)# remove right and top spines
plt.gca().spines['left'].set_visible(False)

# remove axes
#plt.gca().set_yticklabels([])
#plt.gca().set_yticks([])

# add a smooth grid
plt.grid(True, linestyle='--', alpha=0.5)


# Show plot
plt.savefig("plotting_AvsC_italiancities/plots/AvsC_italiancities_income", dpi=600)
plt.show()

