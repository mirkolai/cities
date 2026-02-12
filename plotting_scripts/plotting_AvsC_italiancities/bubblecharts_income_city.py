import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

# Load the JSON data
data = json.load(open("json/income italian cities neigbourhoods.json"))

# Extract data for plotting
cities = list(data.keys())

# select only italian cities
italian_cities = [city for city in cities if city.split(",")[-1].strip() == "Italy"]
# cities_to_exclude = ["Catania, Italy"]

# exclude cities
#for city in cities_to_exclude:
#    italian_cities.remove(city)

for city in italian_cities:
    city_name = city.split(",")[0]

    print(city_name)

    neighbourhoods = data[city]

    # assign to every city a color based on the region
    colors = np.array([neighbourhoods[neig]["color"] for neig in neighbourhoods])
    connectivity = np.array([neighbourhoods[neig]["connectivity_norm"] for neig in neighbourhoods])
    accessibility = np.array([neighbourhoods[neig]["Accessibility_norm"] for neig in neighbourhoods])
    income = np.array([neighbourhoods[neig]["income"] for neig in neighbourhoods])
    neighbourhoods_names = np.array([neig.split(".")[0] for neig in
                                    neighbourhoods])

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
    #plt.title(f"{city_name} - Income")
    plt.gca().yaxis.set_label_position("right")
    plt.gca().yaxis.tick_right()

    # setting x and y axis limits
    plt.xlim(min(connectivity)-min(connectivity)*0.15, max(connectivity)+max(connectivity)*0.15)
    plt.ylim(min(accessibility)-min(accessibility)*0.15, max(accessibility)+max(accessibility)*0.15)

    # add a smooth grid
    plt.grid(True, linestyle='--', alpha=0.5)

    # Plot population legend in lower right corner
    scaler = MinMaxScaler(feature_range=(100, 1000))
    sizes = scaler.fit_transform(np.array([min(income), np.mean(income), max(income)]).reshape(-1, 1)).flatten()
    incsizes = zip([min(income), np.mean(income), max(income)], sizes)
    for income, size in incsizes:
        plt.scatter([], [], edgecolors='k', facecolors='none', s=size, label=f'{income/1000:.2f} k€')


    legend1 = plt.legend(title="Income", scatterpoints=1, labelspacing=0.5, loc='upper right', handletextpad=1.5, frameon=False)
    plt.gca().add_artist(legend1)  # Keep the first legend

    # remove right and top spines
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # Show plot
    plt.savefig(f"plotting_AvsC_italiancities/plots/AvsC_{city_name}_income", dpi=600)
    #plt.show()

