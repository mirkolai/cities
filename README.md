## Article
This repository contains the source code used for the study presented in the article:

**Authors:** Mirko Lai, Anna Sapienza, Salvatore Vilella, Massimo Canonico, Federica Cena, Giancarlo Ruffo

**Title:** *Understanding the interplay between urban segregation and accessibility to services with network analysis*  

**Abstract:** This paper proposes to exploit complex network analysis to evaluate walkable accessibility to services in cities, and how this relates to connectivity provided by the urban transport system. The main research question is whether the hyper-proximity paradigm, of which the popular 15-minute city is an instance, plays a role in urban segregation, increasing the risk of ghettoization and social exclusion in cities and neighborhoods that adhere to this model. We present an analytical approach that includes the quantification of both accessibility to services and urban segregation and allows for detailed comparisons between different cities and different urban areas. The concept of hyper-proximity emphasizes the importance of having basic services and amenities within a few minutes' walk or bike ride to enhance urban livability and sustainability. By modeling cities as complex networks, we analyze various metrics to capture the relationships and accessibility patterns within urban spaces at different scales: residential addresses, network clusters (in place of administrative neighborhoods), and whole cities. Our methodology evaluates the distribution of relevant Points of Interests (PoIs) and how the urban transport network efficiently connects residential areas within a city. In particular, we suggest using closeness centrality as a metric to quantify urban segregation. We found out that accessibility to services and closeness are at interplay, and the co-existence of apparently contradictory patterns emerges world wide: in fact, we observe that areas with poor/good accessibility to services tend to exhibit poor/good urban transport connectivity at different scales, but also that the fraction of citizens who are poorly served is not equally distributed worldwide. Altogether, for Italian cities for which income data were available, we show how disparities in access to essential services can often be interpreted in terms of other socio-economic inequalities so that poorer accessibility and connectivity are likely to be manifested in lower income' neighborhoods. More interestingly, we observed some outliers suggesting that higher income neighborhoods may eventually tend towards voluntary isolation.



## Data Availability

The dataset used for the analysis is **not included** in this repository. However, all data sources are freely available for download from the following platforms:

- **OSM (OpenStreetMap)** – for maps and Points of Interest (PoIs)  
- **WorldPop** – for population density data  
- **Transitland** – for GTFS transit feeds  
  - For Italian cities and Istanbul, we have manually compiled a list of the open data links providing GTFS datasets we used in the analysis.  

Although the raw data is not provided here, we have included the necessary code to **automate the data retrieval process**, ensuring full reproducibility of our analysis pipeline.

## Reproducibility

By running the provided scripts, users can download and preprocess the required datasets to replicate the entire analysis workflow

# Script Execution Order  

The scripts must be executed in the following order:  

## 1. `_compute_PoI_accessibility.py`  

This script requires the most computational resources in terms of execution time and memory.  

### **Operations performed:**  
- Checks if the city has already been processed.  
- Loads population data from WorldPop.  
- Builds the city graph from OpenStreetMap.  
- Extends the graph for better isochrone computations.  
- Extracts Points of Interest (PoIs) from the extended graph.  
- Uses parallel processing to compute accessibility metrics in batches.  


### **Generated files (per city):**  
- `data/worldpop/{country}` – A raster file from WorldPop with population estimates (downloaded once per country and used for all cities in that nation).  
- `output/{place_name}.graphml.gz` – The city's road network extracted from OpenStreetMap (OSM).  
- `output/{place_name} extended.graphml.gz` – An expanded version of the city's bounding box, covering the maximum travel distance reachable within the set time and speed constraints.  
- `output/{place_name}.csv.gz` – A file containing Points of Interest (PoIs) retrieved from OSM.  

## 2. `_recover_GTFS_feeds.py`  

Retrieves GTFS feeds using the Transit.land API.  
### **Generated files (per city):**  
- `data/GTFS/Feeds/{place_name}.json}` dictionary containing the list of feeds recovered by transit.land

> **Note:** Registration on the platform and obtaining an API key are required.  

## 3. `_download_GTFS_from_feed.py`  

Downloads GTFS files from the feeds obtained in step 2.  
### **Generated files (per city):**  
- `data/GTFS/{feed_id} {place_name}.zip` list of GTFS zips for each feed

## 4. `_create_public_transit_graph.py`  

Integrates public transport data (GTFS) into the road network.  
- Produces the file: `{place_name} - walk & transit.graphml.gz`  
- This network is compatible with **networkx** and **osmnx**.  

## 5. `_compute_pop.py`  

Since WorldPop TIFF files provide population density in 100-meter cells, this script:  
- Determines how many intersections fall within each cell.  
- Distributes the population equally among the intersections.  

### **Generated file:**  
- `output/{place_name} Pop.json.gz` – A dictionary with:  
  - **Key:** Node ID  
  - **Value:** Estimated population  

## 6. `_compute_infomap.py`  

Generates clusters within the road network.  

### **Generated files:**  
- `output/{place_name} infomap.json.gz` – A dictionary with:  
  - **Key:** Node ID  
  - **Value:** Cluster ID  
- `output/{place_name} infomap.pickle` – A dictionary with:  
  - **Key:** Cluster ID  
  - **Value:** Convex and concave cluster polygons  

## 7. `_compute_closeness.py`  

Computes **closeness centrality** for all intersections in the combined road and public transport network.  

### **Generated file:**  
- `output/{place_name} closeness.json.gz` – A dictionary with:  
  - **Key:** Node ID  
  - **Value:** Closeness centrality score  

---

Each script must be executed in sequence to ensure full reproducibility of the analysis. 🚀  



If future researchers wish to build upon our results, we have included **preprocessed data** used to create scatter plots and rankings, ensuring that key insights remain accessible.