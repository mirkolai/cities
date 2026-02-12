# Cities bubble charts — Scripts Overview

This repository contains Python scripts to generate bubble charts, maps and correlation matrices from the JSON datasets in the `json/` folder. The README below describes every script outside directories whose names start with `_` or `.`. Files inside `_`-prefixed directories were intentionally ignored.

## Summary of scripts

- **[plotting_AvsC/bubblecharts_AvsC.py](plotting_AvsC/bubblecharts_AvsC.py)**: Loads `json/popolazione_aggregata.json`, computes correlations (Pearson, Kendall) between city `connectivity_norm` and `accessibility_norm`, detects outliers (OneClassSVM / IsolationForest / EllipticEnvelope) and produces a global bubble chart of connectivity vs accessibility. Saves plot to `plotting_AvsC/plots/AvsC_allcities`.

- **[plotting_AvsC/bubblecharts_by_region.py](plotting_AvsC/bubblecharts_by_region.py)**: Provides `plot_bubble_chart_by_region(selected_region=None, selected_countries=None)` to create region- or country-filtered bubble charts (connectivity vs accessibility). Computes correlations and saves per-region plots to `plotting_AvsC/plots/AvsC_<region>` when run.

- **[plotting_AvsC/bubblechart_population_city.py](plotting_AvsC/bubblechart_population_city.py)**: Reads `json/population cities neigbourhoods.json` and, for each city, computes neighborhood-level statistics (Pearson/Kendall, means, medians) and generates a bubble chart per city (connectivity vs accessibility). Saves per-city plots to `plotting_AvsC/plots/AvsC_<city>` and writes `plotting_AvsC/statistics.csv` with summary measures.

- **[plotting_DvsC/bubblecharts_DvsC.py](plotting_DvsC/bubblecharts_DvsC.py)**: Loads `json/popolazione_aggregata_2.json`, computes correlations between `connectivity` and PoI `density`, detects outliers, and draws a global bubble chart of connectivity vs density. Saves output to `plotting_DvsC/plots/DvsC_allcities`.

- **[plotting_EvsC/bubblecharts_EvsC.py](plotting_EvsC/bubblecharts_EvsC.py)**: Loads `json/popolazione_aggregata_2.json`, computes correlations between `connectivity` and PoI `entropy`, detects outliers, and draws a global bubble chart of connectivity vs entropy (visualization via matplotlib).

- **[plotting_PvsC/bubblecharts_PvsC.py](plotting_PvsC/bubblecharts_PvsC.py)**: Loads `json/popolazione_aggregata_2.json`, computes correlations between `connectivity` and PoI `proximity`, detects outliers, and draws a global bubble chart of connectivity vs proximity.

- **[plotting_PvsC/scatter.py](plotting_PvsC/scatter.py)**: Loads `json/scatter-{city_name}.json`, computes correlations between `connectivity` and PoI `proximity`, and draws a scatter plot of connectivity vs proximity scaled by population.

- **[plotting_AvsC_italiancities/bubblecharts_income_city.py](plotting_AvsC_italiancities/bubblecharts_income_city.py)**: Reads `json/income italian cities neigbourhoods.json`, selects Italian cities and for each produces neighborhood-level bubble charts where marker size encodes `income`. Computes correlations and saves per-city income plots to `plotting_AvsC_italiancities/plots/AvsC_<city>_income`.

- **[plotting_AvsC_italiancities/bubblecharts_population_italy.py](plotting_AvsC_italiancities/bubblecharts_population_italy.py)**: Reads `json/income italian cities.json` (contains city-level data), selects Italian cities and plots connectivity vs accessibility for each city aggregated by population; saves `plotting_AvsC_italiancities/plots/AvsC_italiancities_pop`.

- **[plotting_AvsC_italiancities/bubblecharts_income_italy.py](plotting_AvsC_italiancities/bubblecharts_income_italy.py)**: Similar to the population plot but uses `income` as marker size to produce a country-level Italian cities income bubble chart; saves `plotting_AvsC_italiancities/plots/AvsC_italiancities_income`.

- **[italian_map/italy.py](italian_map/italy.py)**: Creates a simple map of Italy (using GeoPandas + Natural Earth base), plots a set of major Italian cities as points sized by population and colored by region (Northern/Central/Southern), annotates city names and saves `italy.png`.

- **[correlation_matrix/correlation_matrix_2.py](correlation_matrix/correlation_matrix_2.py)**: Loads `json/popolazione_aggregata_2.json`, selects measures (connectivity, proximity, density, entropy, accessibility, population), computes a Kendall-tau correlation matrix (and p-values), renders a heatmap annotating significance, and saves `correlation_matrix_kendalltau.png`.

- **[heatmap_accessibility/heatmap_accessibility.py](heatmap_accessibility/heatmap_accessibility.py)**: Loads `json/heatmap_accessibility-{city_name}.json`, renders a heatmap annotating the value of PoI-Accessibility compute for every intersection.

- **[heatmap_closeness/heatmap_closeness.py](heatmap_closeness/heatmap_closeness.py)**: Loads `json/heatmap_closeness-{city_name}.json`, renders a heatmap annotating the value of closeness compute for every intersection.

- **[ranking_proximity/ranking_proximity.py](ranking_proximity/ranking_proximity.py)**: Loads `json/ranking_proximity.json`, creates a box plot to visualize the ranking of cities according to their PoI-Proximity.

- **[ranking_accessibility/ranking_accessibility.py](ranking_accessibility/ranking_accessibility.py)**: Loads `json/ranking_accessibility.json`, creates a box plot to visualize the ranking of cities according to their PoI-Accessibility.

- **[ranking_closeness/ranking_closeness.py](ranking_closeness/ranking_closeness.py)**: Loads `json/ranking_closeness.json`, creates a box plot to visualize the ranking of cities according to their Closeness.

- **[pareto_proximity/pareto_proximity.py](pareto_proximity/pareto_proximity.py)**: Loads `json/pareto_proximity.json`, selects representative cities and plots cumulative PoI-Proximity curves.

- **[accessibility_metrics/accessibility_metrics.py](accessibility_metrics/accessibility_metrics.py)**: Loads `json/accessibility_metrics.json`, plots multiple accessibility metrics (PoI-Proximity, Poi-Density, PoI-Entropy, PoI-Accessibility) distributions for each place visually.

## Data

- All datasets used by the scripts live in the `json/` folder (e.g. `popolazione_aggregata.json`, `popolazione_aggregata_2.json`, `income italian cities.json`, `population cities neigbourhoods.json`, etc.).

## Running

- Most scripts are standalone plotting scripts that can be run with Python 3. They expect the `json/` data files to be present and common packages installed (`numpy`, `matplotlib`, `scipy`, `scikit-learn`, `pandas`, `geopandas`, `seaborn`, `contextily`, `adjustText`, etc.).

Example:

```bash
python3 plotting_AvsC/bubblecharts_AvsC.py
python3 plotting_DvsC/bubblecharts_DvsC.py
python3 italian_map/italy.py
```

## Notes

- Scripts use different JSON files and slightly different field names (e.g. `connectivity_norm` vs `city_connectivity`); check the top of each script for the expected input keys.
- Required Python libraries can be installed from `requirements.txt`.
