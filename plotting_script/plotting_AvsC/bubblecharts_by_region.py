import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import pearsonr
from adjustText import adjust_text

def plot_bubble_chart_by_region(selected_region=None, selected_countries=None):
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
        "New Zealand": "Oceania"
    }

    # Extract data for plotting
    cities = list(data.keys())
    # split the city name to get the city name only, and keep the country name in a separate list
    city_names = np.array([city.split(",")[0].strip() for city in cities])
    country_names = [city.split(",")[-1].strip() for city in cities]

    # assign to every city a region based on the country name
    regions = [country_to_region[country] for country in country_names]

    # highlight_cities = {"Paris, France", "Turin, Italy", "Vancouver, Canada", "Ottawa, Canada", "Melbourne, Australia", "Houston, USA"}
    # highlight italian cities  
    # highlight_cities = {"Turin, Italy", "Milan, Italy", "Rome, Italy", "Naples, Italy", "Palermo, Italy", "Genoa, Italy", "Bologna, Italy", "Florence, Italy", "Bari, Italy", "Catania, Italy"}
    # highlight cities whose country is in North America
    if selected_countries is not None:
        highlight_cities = {city for city, country in zip(cities, country_names) if country in selected_countries}
    else: 
        highlight_cities = {city for city, region in zip(cities, regions) if region == selected_region}
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

    # calculate correlation between connectivity and accessibility
    correlation = np.corrcoef(connectivity, accessibility)[0, 1]
    print("Correlation between connectivity and accessibility:", correlation)

    # calculate pearson correlation coefficient between connectivity and accessibility
    pearson_corr = pearsonr(connectivity, accessibility)[0]
    print("Pearson correlation coefficient between connectivity and accessibility:", pearson_corr)

    # calculate kendall tau correlation coefficient between connectivity and accessibility
    from scipy.stats import kendalltau
    kendall_tau, _ = kendalltau(connectivity, accessibility)
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

    # cities are colored differently based on their country_names

    # Create bubble chart
    plt.figure(figsize=(8, 8))

    # create a list of indices of cities that are not in the highlight_cities
    regular_indices = np.setdiff1d(np.arange(len(cities)), highlight_cities_indices)
    plt.scatter(connectivity[regular_indices], accessibility[regular_indices], color=colors[regular_indices], s=sizes[regular_indices], alpha=0.4,  edgecolors=colors[regular_indices])

    # Add labels for specific cities
    for i in highlight_cities_indices:   
        plt.scatter(connectivity[i], accessibility[i], color=colors[i], s=sizes[i], alpha=0.8, edgecolors='k')


    texts = []
    for i in highlight_cities_indices:
        texts.append(
            plt.text(connectivity[i], accessibility[i], city_names[i], fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.5))
        )

    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='black', lw=0.8))
    '''
    for i in highlight_cities_indices:
        xshift = 0.015
        yshift = 0.015
        plt.annotate(city_names[i], 
                    (connectivity[i], accessibility[i]), 
                    xytext=(connectivity[i] + xshift, accessibility[i] + yshift),  # Offset
                    textcoords='data',
                    fontsize=9, 
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.5), 
                    arrowprops=dict(arrowstyle="->", color='black', lw=0.8))
    '''
    # plot y = median_accessibility
    plt.axhline(median_accessibility, color='grey', linestyle='--', alpha=0.5)
    plt.text(0.1, median_accessibility, r"Median $\mathcal{A}$: "+f"{median_accessibility:.2f}", fontsize=9, ha='left', va='bottom')

    #plot x = median_connectivity
    plt.axvline(median_connectivity, color='grey', linestyle='--', alpha=0.5)
    plt.text(median_connectivity, 0.2, r"Median $\mathcal{C}$: "+f"{median_connectivity:.2f}", fontsize=9, ha='left', va='bottom')

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


    # adding a legend on region
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
    plt.savefig("plotting_AvsC/plots/AvsC_"+selected_region, transparent=False, dpi=600)
    #plt.show()

if __name__ == "__main__":
    # plot bubble chart for cities in Europe
    plot_bubble_chart_by_region(selected_region="Europe")
    # plot bubble chart for cities in North America
    plot_bubble_chart_by_region(selected_region="North America")
    # plot bubble chart for cities in Latin America and Carribean    
    plot_bubble_chart_by_region(selected_region="Latin America and Carribean")
    # plot bubble chart for cities in Africa
    plot_bubble_chart_by_region(selected_region="Africa")
    # plot bubble chart for cities in Asia
    plot_bubble_chart_by_region(selected_region="Asia")
    # plot bubble chart for cities in Oceania
    plot_bubble_chart_by_region(selected_region="Oceania")
    

    # uncomment one of the following lines to plot bubble chart for cities in selected_countries
    # selected_countries = ["Denmark", "Ireland", "UK", "Norway", "Finland", "Sweden", "Estonia"] # select_countries is a list of countries in Northern Europe
    # selected_countries = ["Netherlands", "Germany", "France", "Austria", "Switzerland"] # select_countries is a list of countries in Western Europe
    # selected_countries = ["Spain", "Italy", "Portugal", "Greece", "Turkey"] # select_countries is a list of countries in Southern Europe
    # selected_countries = ["Hungary", "Czechia", "Poland"]  # select_countries is a list of countries in Eastern Europe 
    # selected_countries = ["Canada", "Mexico"]  # selected_countries is a list of countries in North America, without USA
    # selected_countries = ["USA"]

    # plot bubble chart for cities in selected_countries 
    # uncomment the following line to plot bubble chart for cities in selected_countries
    # plot_bubble_chart_by_region(selected_countries=selected_countries)
