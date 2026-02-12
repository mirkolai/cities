import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

# Load the JSON data
data = json.load(open("json/population cities neigbourhoods.json"))

# Extract data for plotting
cities = list(data.keys())

# select only italian cities
#italian_cities = [city for city in cities if city.split(",")[-1].strip() == "Italy"]

import pandas as pd
measures = pd.DataFrame(columns=['city_name', 'pearson_corr', 'pearson_pvalue', 'kendall_tau', 'kendall_tau_pvalue', 'mean_connectivity', 'std_connectivity', 'mean_accessibility', 'std_accessibility', 'median_connectivity', 'median_accessibility'])

for city in cities:
    city_name = city.split(",")[0]

    print(city_name)
    # create a new row in the dataframe

    neighbourhoods = data[city]

    # assign to every city a color based on the region
    colors = np.array([neighbourhoods[neig]["color"] for neig in neighbourhoods])
    connectivity = np.array([neighbourhoods[neig]["connectivity_norm"] for neig in neighbourhoods])
    accessibility = np.array([neighbourhoods[neig]["Accessibility_norm"] for neig in neighbourhoods])
    population = np.array([neighbourhoods[neig]["population"] for neig in neighbourhoods])
    neighbourhoods_names = np.array([neig.split(".")[0] for neig in
                                    neighbourhoods])

    # Normalize marker size
    scaler = MinMaxScaler(feature_range=(100, 1000))
    sizes = scaler.fit_transform(population.reshape(-1, 1)).flatten()

    # calculate correlation between connectivity and accessibility
    correlation = np.corrcoef(connectivity, accessibility)[0, 1]
    print("Correlation between connectivity and accessibility:", correlation)

    # calculate pearson correlation coefficient between connectivity and accessibility
    pearson_corr,pearson_pvalue = pearsonr(connectivity, accessibility)
    print("Pearson correlation coefficient between connectivity and accessibility:", pearson_corr)


    # calculate kendall tau correlation coefficient between connectivity and accessibility
    from scipy.stats import kendalltau
    kendall_tau, kendall_tau_pvalue = kendalltau(connectivity, accessibility)
    print("Kendall tau correlation coefficient between connectivity and accessibility:", kendall_tau)


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

    measures.loc[len(measures)] = {'city_name': city_name, 
                                'pearson_corr': pearson_corr, 'pearson_pvalue': pearson_pvalue, 
                                'kendall_tau': kendall_tau, 'kendall_tau_pvalue': kendall_tau_pvalue, 
                                'mean_connectivity': mean_connectivity, 'std_connectivity': std_connectivity, 
                                'mean_accessibility': mean_accessibility, 'std_accessibility': std_accessibility, 
                                'median_connectivity': median_connectivity, 'median_accessibility': median_accessibility}

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
    neighbourhoods_names_sorted = [neighbourhoods_names[i] for i in sorted_indices]

    plt.scatter(connectivity_sorted, accessibility_sorted, s=sizes_sorted, color=colors_sorted, edgecolors='k', alpha=0.5)

    # Annotare i punti con frecce e sfondo per le etichette
    for i in range(len(connectivity_sorted)):
        xshift = 0.01
        yshift = 0.01
        plt.annotate(neighbourhoods_names_sorted[i], 
                    (connectivity_sorted[i], accessibility_sorted[i]), 
                    xytext=(connectivity_sorted[i] + xshift, accessibility_sorted[i] + yshift),  # Offset
                    textcoords='data',
                    fontsize=9, 
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'), 
                    arrowprops=dict(arrowstyle="->", color='black', lw=0.8))

    # Labels and Titles
    plt.xlabel("Normalized closeness")
    plt.ylabel("PoI-accessibility")
    #plt.title(f"{city_name} - Population")

    # setting x and y axis limits
    plt.xlim(min(connectivity)-min(connectivity)*0.15, max(connectivity)+max(connectivity)*0.15)
    plt.ylim(min(accessibility)-min(accessibility)*0.15, max(accessibility)+max(accessibility)*0.15)

    # add a smooth grid
    plt.grid(True, linestyle='--', alpha=0.5)

    # Plot population legend in lower right corner
    scaler = MinMaxScaler(feature_range=(100, 1000))
    sizes = scaler.fit_transform(np.array([min(population), np.mean(population), max(population)]).reshape(-1, 1)).flatten()
    popsizes = zip([min(population), np.mean(population), max(population)], sizes)
    for pop, size in popsizes:
        plt.scatter([], [], edgecolors='k', facecolors='none', s=size, label=f'{pop/1000:.2f} k')


    legend1 = plt.legend(title="Population", scatterpoints=1, labelspacing=0.5, loc='upper left', handletextpad=1.5, frameon=False)
    plt.gca().add_artist(legend1)  # Keep the first legend

    # remove right and top spines
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # Show plot
    plt.savefig(f"plotting_AvsC/plots/AvsC_{city_name}", dpi=600)
    #plt.show()

measures.to_csv("plotting_AvsC/statistics.csv", index=False)