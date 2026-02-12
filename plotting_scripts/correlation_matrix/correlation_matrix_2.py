import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import kendalltau
import seaborn as sns

def plot_correlation_matrix(correlation_function=kendalltau, correlation_string="kendalltau"):
    # Load the JSON data
    data = json.load(open("json/popolazione_aggregata_2.json"))

    country_to_region = {
        "USA": "North America",
        "Canada": "North America",
        "Mexico": "North America",
        "Argentina": "Latin American and Carribean",
        "Brazil": "Latin American and Carribean",
        "Colombia": "Latin American and Carribean",
        "Chile": "Latin American and Carribean",
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
        "Perù": "Latin American and Carribean",
        "Venezuela": "Latin American and Carribean",
        "Bolivia": "Latin American and Carribean",
        "Paraguay": "Latin American and Carribean",
        "Uruguay": "Latin American and Carribean",
        "Cuba": "Latin American and Carribean",
        "Jamaica": "Latin American and Carribean",
        "Haiti": "Latin American and Carribean",
        "Dominican Republic": "Latin American and Carribean",
        "Puerto Rico": "Latin American and Carribean",
        "Trinidad and Tobago": "Latin American and Carribean",
        "Barbados": "Latin American and Carribean",
        "Saint Lucia": "Latin American and Carribean",
        "Saint Vincent and the Grenadines": "Latin American and Carribean",
        "Grenada": "Latin American and Carribean",
        "Antigua and Barbuda": "Latin American and Carribean",
        "Saint Kitts and Nevis": "Latin American and Carribean",
        "Bahamas": "Latin American and Carribean",
        "Belize": "Latin American and Carribean",
        "Guatemala": "Latin American and Carribean",
        "El Salvador": "Latin American and Carribean",
        "Honduras": "Latin American and Carribean",
        "Nicaragua": "Latin American and Carribean",
        "Costa Rica": "Latin American and Carribean",
        "Panama": "Latin American and Carribean",
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
    "Latin American and Carribean": "#ff7f00",#
    "Europe": "#984ea3",#
    "Africa": "#4daf4a",#
    "Asia": "#377eb8" ,#
    "Oceania": "#e41a1c",#
    }

    # assign to every city a color based on the region
    colors = np.array([continent_to_color[region] for region in regions])

    connectivity = np.array([city["city_connectivity"] for city in data.values()])
    accessibility = np.array([city["city_accessibility"] for city in data.values()])
    proximity = np.array([city["city_proximity"] for city in data.values()])
    population = np.array([city["city_population"] for city in data.values()])
    proximity = np.array([city["city_proximity"] for city in data.values()])
    density = np.array([city["city_density"] for city in data.values()])
    entropy = np.array([city["city_entropy"] for city in data.values()])
    # Normalize marker size
    sizes = population / max(population) * 2000  # Scale for visibility

    # keep only cities with connectivity greater than 0
    valid_indices = np.where(connectivity > 0)[0]
    city_names = city_names[valid_indices]
    colors = colors[valid_indices]
    sizes = sizes[valid_indices]
    connectivity = connectivity[valid_indices]
    accessibility = accessibility[valid_indices]
    proximity = proximity[valid_indices]
    population = population[valid_indices]      
    density = density[valid_indices]
    entropy = entropy[valid_indices]    


    # Calculate the kendall-tau correlation matrix
    # Compute the full 6x6 correlation matrix for all pairs
    measures = [connectivity, proximity, density, entropy, accessibility, population]
    num_measures = len(measures)

    variables = [r'$\mathcal{C}$', r'$\mathcal{P}$', r'$\mathcal{D}$', r'$\mathcal{E}$', r'$\mathcal{A}$', 'Pop']
    variables_names = ['Closeness', 'PoI-proximity', 'PoI-density', 'PoI-entropy', 'PoI-accessibility', 'Population']

    # Creazione della matrice vuota per tau e p-value
    correlation_matrix = np.zeros((num_measures, num_measures))
    pvalues_matrix = np.ones((num_measures, num_measures))
    for i in range(num_measures):
        for j in range(num_measures):
            #correlation_matrix[i, j] = pearsonr(measures[i], measures[j])[0]
            #pvalues_matrix[i, j] = pearsonr(measures[i], measures[j])[1]
            correlation_matrix[i, j] = correlation_function(measures[i], measures[j])[0]
            pvalues_matrix[i, j] = correlation_function(measures[i], measures[j])[1]

    # Creazione della maschera per nascondere il triangolo inferiore e la diagonale
    mask = np.tril(np.ones_like(correlation_matrix, dtype=bool))  # k=0 include la diagonale

    # Creazione della heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5, square=True,
                mask=mask, cbar_kws={"shrink": 0.5}, xticklabels=False, yticklabels=True, annot_kws={"color": "black"})

    #plt.xticks(np.arange(len(variables)), variables, rotation=45)
    plt.yticks(np.arange(len(variables)), variables) 

    # plt.title("Pearson Correlation Matrix", fontsize=11, pad=10)  

    # Spostare le etichette della x in alto
    #ax.xaxis.tick_top()
    #ax.xaxis.set_label_position('top')
    #plt.xticks(rotation=45, ha='left')

    # non mostrare le etichette dell'asse X
    ax.set_xticklabels([])

    # Rimozione delle tacche dell'asse Y
    ax.set_yticks([])
    ax.tick_params(axis='y', bottom=False, top=False)
    ax.tick_params(axis='x', bottom=False, top=False)



    # Aggiungere i p-value sopra i quadrati della heatmap
    for i in range(len(variables)):
        for j in range(len(variables)):
            if i < j:  # Solo triangolo superiore
                p_val = pvalues_matrix[i, j]
                if p_val < 0.05:  # Mostra solo p-value significativi
                    text_color = "black"
                    ax.text(j + 0.5, i + 0.17, f"p < 0.05", ha='center', va='center', fontsize=8, color=text_color)
                else:
                    text_color = "black"
                    ax.text(j + 0.5, i + 0.17, f"p ≥ 0.05", ha='center', va='center', fontsize=8, color=text_color)

    # Scrivi le etichette delle variabili sulla diagonale (al posto dei valori)
    for i, label in enumerate(variables):
        ax.text(i + 0.5, i + 0.3, variables_names[i], ha='center', va='center', fontsize=5, color="black")
        ax.text(i + 0.5, i + 0.6, label, ha='center', va='center', fontsize=11, color="black")

    # Rimuovi le yticklabels e xticklabels per evitare doppioni
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # Spazio regolato
    plt.subplots_adjust(left=0.2, right=0.9, top=0.823, bottom=0.2)

    # Salvataggio e visualizzazione
    plt.savefig(f"correlation_matrix/plots/correlation_matrix_{correlation_string}.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_correlation_matrix(correlation_function=kendalltau, correlation_string="kendalltau")   
    plot_correlation_matrix(correlation_function=pearsonr, correlation_string="pearson")