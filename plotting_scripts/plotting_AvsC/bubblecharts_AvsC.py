import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import pearsonr

# Load the JSON data
data = json.load(open("json/popolazione_aggregata.json"))

country_to_region = {
    "USA": "North America",
    "Canada": "North America",
    "Mexico": "North America",
    "Argentina": "Latin America and Carribean",
    "Brazil": "Latin America and Carribean",
    "Colombia": "Latin America and Carribean",
    "Chile": "Latin America and Carribean",
    "Spain": "Europe",
    "Italy": "Europe",
    "Netherlands": "Europe",
    "Germany": "Europe",
    "Hungary": "Europe",
    "Czechia": "Europe",
    "Denmark": "Europe",
    "Ireland": "Europe",
    "UK": "Europe",
    "Portugal": "Europe",
    "Greece": "Europe",
    "Norway": "Europe",
    "Finland": "Europe",
    "France": "Europe",
    "Turkey": "Europe",
    "Sweden": "Europe",
    "Austria": "Europe",
    "Poland": "Europe",
    "Switzerland": "Europe",
    "Ethiopia": "Africa",
    "Kenya": "Africa",
    "Japan": "Asia",
    "Thailand": "Asia",
    "Indonesia": "Asia",
    "Philippines": "Asia",
    "Singapore": "Asia",
    "Taiwan": "Asia",
    "Estonia": "Europe",
    "Australia": "Oceania",
    "New Zealand": "Oceania",
    "China": "Asia",
    "South Korea": "Asia",
    "Russia": "Europe",
    "India": "Asia",
    "Pakistan": "Asia",
    "Bangladesh": "Asia",
    "Vietnam": "Asia",
    "Malaysia": "Asia",
    "Hong Kong": "Asia",
    "United Arab Emirates": "Asia",
    "Saudi Arabia": "Asia",
    "Qatar": "Asia",
    "South Africa": "Africa",
    "Nigeria": "Africa",
    "Morocco": "Africa",
    "Egypt": "Africa",
    "Tunisia": "Africa",
    "Algeria": "Africa",
    "Senegal": "Africa",
    "Ghana": "Africa",
    "Cameroon": "Africa",
    "Ivory Coast": "Africa",
    "Tanzania": "Africa",
    "Uganda": "Africa",
    "Zimbabwe": "Africa",
    "Zambia": "Africa",
    "Rwanda": "Africa",
    "Burundi": "Africa",
    "Malawi": "Africa",
    "Mozambique": "Africa",
    "Angola": "Africa",
    "Democratic Republic of the Congo": "Africa",
    "Republic of the Congo": "Africa",
    "Namibia": "Africa",
    "Botswana": "Africa",
    "Lesotho": "Africa",
    "Swaziland": "Africa",
    "Seychelles": "Africa",
    "Mauritius": "Africa",
    "Madagascar": "Africa",
    "Cape Verde": "Africa",
    "Gambia": "Africa",
    "Liberia": "Africa",
    "Sierra Leone": "Africa",
    "Guinea": "Africa",
    "Guinea-Bissau": "Africa",
    "Mali": "Africa",
    "Niger": "Africa",
    "Chad": "Africa",
    "Central African Republic": "Africa",
    "South Sudan": "Africa",
    "Sudan": "Africa",
    "Somalia": "Africa",
    "Eritrea": "Africa",
    "Djibouti": "Africa",
    "Comoros": "Africa",
    "Sao Tome and Principe": "Africa",
    "Togo": "Africa",
    "Benin": "Africa",
    "Burkina Faso": "Africa",
    "Mauritania": "Africa",
    "Gabon": "Africa",
    "Equatorial Guinea": "Africa",
    "Congo": "Africa",
    "Libya": "Africa",
    "Syria": "Asia",
    "Iraq": "Asia",
    "Lebanon": "Asia",
    "Jordan": "Asia",   
    "Oman": "Asia",
    "Kuwait": "Asia",
    "Bahrain": "Asia",
    "Yemen": "Asia",
    "Afghanistan": "Asia",
    "Kazakhstan": "Asia",
    "Uzbekistan": "Asia",
    "Turkmenistan": "Asia",
    "Kyrgyzstan": "Asia",
    "Tajikistan": "Asia",
    "Armenia": "Asia",
    "Azerbaijan": "Asia",
    "Georgia": "Asia",
    "Sri Lanka": "Asia",
    "Nepal": "Asia",
    "Bhutan": "Asia",
    "Maldives": "Asia",
    "Myanmar": "Asia",
    "Cambodia": "Asia",
    "Laos": "Asia",
    "Brunei": "Asia",
    "Perù": "Latin America and Carribean",
    "Venezuela": "Latin America and Carribean",
    "Bolivia": "Latin America and Carribean",
    "Paraguay": "Latin America and Carribean",
    "Uruguay": "Latin America and Carribean",
    "Cuba": "Latin America and Carribean",
    "Jamaica": "Latin America and Carribean",
    "Haiti": "Latin America and Carribean",
    "Dominican Republic": "Latin America and Carribean",
    "Puerto Rico": "Latin America and Carribean",
    "Trinidad and Tobago": "Latin America and Carribean",
    "Barbados": "Latin America and Carribean",
    "Saint Lucia": "Latin America and Carribean",
    "Saint Vincent and the Grenadines": "Latin America and Carribean",
    "Grenada": "Latin America and Carribean",
    "Antigua and Barbuda": "Latin America and Carribean",
    "Saint Kitts and Nevis": "Latin America and Carribean",
    "Bahamas": "Latin America and Carribean",
    "Belize": "Latin America and Carribean",
    "Guatemala": "Latin America and Carribean",
    "El Salvador": "Latin America and Carribean",
    "Honduras": "Latin America and Carribean",
    "Nicaragua": "Latin America and Carribean",
    "Costa Rica": "Latin America and Carribean",
    "Panama": "Latin America and Carribean",
    "Bermuda": "North America",
    "Greenland": "North America",
    "Iceland": "Europe",
    "Cyprus": "Europe",
    "Malta": "Europe",
    "Andorra": "Europe",
    "Liechtenstein": "Europe",
    "Monaco": "Europe",
    "San Marino": "Europe",
    "Vatican City": "Europe",
    "Moldova": "Europe",
    "Belarus": "Europe",
    "Ukraine": "Europe"
}

# Extract data for plotting
cities = list(data.keys())
# split the city name to get the city name only, and keep the country name in a separate list
city_names = np.array([city.split(",")[0].strip() for city in cities])
country_names = [city.split(",")[-1].strip() for city in cities]

# assign to every city a region based on the country name
regions = [country_to_region[country] for country in country_names]

highlight_cities = {"Paris, France", "Turin, Italy", "Vancouver, Canada", "Ottawa, Canada", "Melbourne, Australia", "Houston, USA"}
# highlight italian cities  
# highlight_cities = {"Turin, Italy", "Milan, Italy", "Rome, Italy", "Naples, Italy", "Palermo, Italy", "Genoa, Italy", "Bologna, Italy", "Florence, Italy", "Bari, Italy", "Catania, Italy"}

# get the indices of the highlight cities   
highlight_cities_indices = np.array([i for i, city in enumerate(cities) if city in highlight_cities])


continent_to_color={
"North America": "#FFD700",#
"Latin America and Carribean": "#ff7f00",#
"Europe": "#984ea3",#
"Africa": "#4daf4a",#
"Asia": "#377eb8" ,#
"Oceania": "#e41a1c",#
}

# assign to every city a color based on the region
colors = np.array([continent_to_color[region] for region in regions])

connectivity = np.array([city["connectivity_norm"] for city in data.values()])
accessibility = np.array([city["accessibility_norm"] for city in data.values()])
population = np.array([city["population"] for city in data.values()])
# Normalize marker size
sizes = population / max(population) * 2000  # Scale for visibility

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

# print mean and standard deviation for each city
print("Mean connectivity:", mean_connectivity)
print("Standard deviation connectivity:", std_connectivity)
print("Mean accessibility:", mean_accessibility)
print("Standard deviation accessibility:", std_accessibility)
print("Median connectivity:", median_connectivity)
print("Median accessibility:", median_accessibility)

X = np.column_stack((connectivity, accessibility)) # create a 2D array with connectivity and accessibility


# use one class svm to identify outliers
from sklearn.svm import OneClassSVM
clf = OneClassSVM(gamma='auto').fit(X)
y_pred = clf.predict(X)
outliers = X[y_pred == -1]
print("Outliers:", outliers)    


# use isolation forest to identify outliers
from sklearn.ensemble import IsolationForest
clf = IsolationForest(contamination=0.1)
y_pred = clf.fit_predict(X)
outliers = X[y_pred == -1]
print("Outliers:", outliers)

# use elliptic envelope to identify outliers
from sklearn.covariance import EllipticEnvelope
clf = EllipticEnvelope(contamination=0.22)
y_pred = clf.fit_predict(X)
outliers = X[y_pred == -1]
print("Outliers:", outliers)    


# calculate pearson correlation coefficient between connectivity and accessibility
pearson_corr, pearson_pvalue = pearsonr(connectivity, accessibility)
print("Pearson correlation coefficient between connectivity and accessibility:", pearson_corr)
print("Pearson correlation p-value between connectivity and accessibility:", pearson_pvalue)

# calculate kendall tau correlation coefficient between connectivity and accessibility
from scipy.stats import kendalltau
kendall_tau,  kendall_tau_pvalue = kendalltau(connectivity, accessibility)
print("Kendall tau correlation coefficient between connectivity and accessibility:", kendall_tau)
print("Kendall tau correlation p-value between connectivity and accessibility:", kendall_tau_pvalue)


# cities are colored differently based on their country_names
import matplotlib.pyplot as plt
import numpy as np

# Create bubble chart
plt.figure(figsize=(8, 8))


# Plot outliers in red
outlier_indices = np.where(y_pred == -1)
# remove highlight cities from outlier indices
outlier_indices = np.setdiff1d(outlier_indices, highlight_cities_indices)
plt.scatter(connectivity[outlier_indices], accessibility[outlier_indices], color=colors[outlier_indices], s=sizes[outlier_indices], hatch="++", alpha=0.4, edgecolors=colors[outlier_indices])
edgecolors = np.array(colors)  # Initialize edgecolors with the same colors
alpha = np.array([0.2] * len(colors))  # Initialize alpha with the same value for all points
for i in outlier_indices:
    xshift = 0.015
    yshift = 0.015
    edgecolors[i] = 'grey'
    if city_names[i] == "Manchester":
        xshift = -0.1
        yshift = -0.015
    if city_names[i] == "Bogota":
        xshift = -0.1
        yshift = -0.015
    if city_names[i] == "Jakarta":
        xshift = 0.03
        yshift = -0.015
    if city_names[i] in ['Buenos Aires', 'Bogota', 'Adelaide', 'Barcelona', 'Rio de Janeiro', 'Jakarta']:
        colors[i] = 'white'
        edgecolors[i] = 'black'
        alpha[i] = 1

    plt.annotate(city_names[i], 
                 (connectivity[i], accessibility[i]), 
                 xytext=(connectivity[i] + xshift, accessibility[i] + yshift),  # Offset
                 textcoords='data',
                 fontsize=9, 
                 color=edgecolors[i],
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor=edgecolors[i], facecolor=colors[i], alpha=alpha[i]), 
                 arrowprops=dict(arrowstyle="->", color='black', lw=0.8))
# Plot all points in blue
regular_indices = np.where(y_pred == 1)
# remove highlight cities from regular indices
regular_indices = np.setdiff1d(regular_indices, highlight_cities_indices)
plt.scatter(connectivity[regular_indices], accessibility[regular_indices], color=colors[regular_indices], s=sizes[regular_indices], alpha=0.4,  edgecolors=colors[regular_indices])
#for i in regular_indices:
#   plt.text(connectivity[i] + np.log(sizes[i])/400, accessibility[i], city_names[i], fontsize=8, ha='left', va='center')


# Annotate points
#for i, label in enumerate(labels):
#    plt.text(x[i], y[i], label, fontsize=9, ha='right', va='bottom')

# for each city, plot the country name
#for i, city in enumerate(cities):
#    plt.text(connectivity[i], accessibility[i], country_names[i], fontsize=10, ha='right', va='bottom')

# Add labels for specific cities
for i in highlight_cities_indices:
    # if i in outlier_indices, plot with a different marker
    if i in np.where(y_pred == -1)[0]:
        plt.scatter(connectivity[i], accessibility[i], color=colors[i], s=sizes[i], alpha=0.8, edgecolors='k',hatch="++")
    else:
        plt.scatter(connectivity[i], accessibility[i], color=colors[i], s=sizes[i], alpha=0.8, edgecolors='k')

for i in highlight_cities_indices:
    xshift = 0.015
    yshift = 0.015
    if city_names[i] == "Copenhagen":
        xshift = 0.018
        yshift = 0.005

    if city_names[i] == "Houston":
        #xshift = -0.1
        yshift = -0.015
    plt.annotate(city_names[i], 
                 (connectivity[i], accessibility[i]), 
                 xytext=(connectivity[i] + xshift, accessibility[i] + yshift),  # Offset
                 textcoords='data',
                 fontsize=9, 
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgray'), 
                 arrowprops=dict(arrowstyle="->", color='black', lw=0.8))

# plot y = median_accessibility
plt.axhline(median_accessibility, color='grey', linestyle='--', alpha=0.5)
plt.text(0.1, median_accessibility, r"Median $\mathcal{A}$: "+f"{median_accessibility:.2f}", fontsize=9, ha='left', va='bottom')

#plot x = median_connectivity
plt.axvline(median_connectivity, color='grey', linestyle='--', alpha=0.5)
plt.text(median_connectivity, 0.2, r"Median $\mathcal{C}$: "+f"{median_connectivity:.2f}", fontsize=9, ha='left', va='bottom')

# plot y = x 
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', alpha=0.5)
plt.text(0.1, 0.1, r"$\mathcal{C} = \mathcal{A}$", fontsize=9, ha='left', va='bottom')

# Labels and Titles
plt.xlabel("Normalized closeness")
plt.ylabel("PoI-accessibility")

# setting x and y axis limits
plt.xlim(0.1, 0.8)
plt.ylim(0.2, 0.8)

# adding a legend on population size
# on a lower right corner    
for size in [1000000, 5000000, 10000000]:
    plt.scatter([], [], edgecolors='k', facecolors='none', s=size/max(population)*2000, label=f'{size/1000000:,} M')
legend1 = plt.legend(title="Population", scatterpoints=1, labelspacing=0.5, loc='lower right', bbox_to_anchor=(1.02,0.03), handletextpad=1.5, frameon=False)
plt.gca().add_artist(legend1)



# adding a legend on continent
# on a lower left corner
handles, labels = plt.gca().get_legend_handles_labels()
for continent in continent_to_color:
    handles.append(plt.Line2D([], [], marker='o', color='w', markerfacecolor=continent_to_color[continent], markersize=10, label=continent))
legend2 = plt.legend(handles=handles[len(handles)-len(continent_to_color):], scatterpoints=1, labelspacing=0.5, loc='lower left', handletextpad=1, frameon=False)
plt.gca().add_artist(legend2)

# add a smooth grid
plt.grid(True, linestyle='--', alpha=0.5)

# remove right and top spines
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

# Show plot
transparent = False
plt.savefig("plotting_AvsC/plots/AvsC_allcities", dpi=600)
plt.show()

