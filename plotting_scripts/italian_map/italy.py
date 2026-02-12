import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np

# Define city data
cities = {
    "Rome": (41.9028, 12.4964, 1721192.4250163091),
    "Florence": (43.7696, 11.2558, 275413.72523861326),
    "Turin": (45.0703, 7.6869, 591465.0282910744),
    "Milan": (45.4642, 9.1900, 848876.7524679189),
    "Catania": (37.5079, 15.0830, 211713.58899587806),
    "Bari": (41.1171, 16.8719, 213419.01851192067),
    "Bologna": (44.4949, 11.3426, 252104.37261033198),
    "Palermo": (38.1157, 13.3615, 390377.280607603),
    "Naples": (40.8518, 14.2681, 592769.6973234285),
    "Genoa": (44.4056, 8.9463, 462891.73953293386),
}

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
areas = [city_to_area[city] for city in cities.keys()]

area_to_color={
"Southern Italy": "#67a9cf",#
"Central Italy": "#f7f7f7",#
"Northern Italy": "#ef8a62"#
}

# assign to every city a color based on the region
colors = np.array([area_to_color[area] for area in areas])

# Convert city data into GeoDataFrame
gdf = gpd.GeoDataFrame(
    {
        "City": list(cities.keys()),
        "Population": [data[2] for data in cities.values()],
        "Size": [data[2]/cities["Rome"][2] * 2000 for data in cities.values()],
    },
    geometry=gpd.points_from_xy([data[1] for data in cities.values()], [data[0] for data in cities.values()]),
    crs="EPSG:4326",
)

# Convert to Web Mercator for plotting
gdf = gdf.to_crs(epsg=3857)

# Load Italy map
gdf_world = gpd.read_file("https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip")
italy = gdf_world[gdf_world["NAME"] == "Italy"].to_crs(epsg=3857)

#regions = gpd.read_file("it_100km.shp").to_crs(epsg=3857)


# Plot
fig, ax = plt.subplots(figsize=(5, 8))
#regions.boundary.plot(ax=ax, color="lightgray")
italy.plot(ax=ax, color="white", edgecolor="black")
gdf.plot(ax=ax, markersize=gdf["Size"], color=colors, edgecolor="black")
#ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)


# Labels
for idx, row in gdf.iterrows():
    xshift = 50000
    yshift = 50000
    #ax.text(row.geometry.x, row.geometry.y, row["City"], fontsize=10, ha="right", color="black")
    plt.annotate(row["City"], 
                 (row.geometry.x, row.geometry.y), 
                 xytext=(row.geometry.x + xshift, row.geometry.y + yshift),  # Offset
                 textcoords='data',
                 fontsize=9, 
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'), 
                 arrowprops=dict(arrowstyle="->", color='black', lw=0.8))

# remove spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# remove axes
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_yticks([])
ax.set_xticks([])
plt.axis('off')

# put all the population values in one list
population = gdf["Population"].tolist()

# plot a different legend for the colors in the upper left corner
handles, labels = plt.gca().get_legend_handles_labels()
for i, area in enumerate(["Northern Italy", "Central Italy", "Southern Italy"]):  
    handles.append(plt.Line2D([], [], marker='o', color='w', markerfacecolor=area_to_color[area], markeredgecolor='k', markersize=10, label=area))
legend2 = plt.legend(handles=handles[len(handles)-len(colors):], scatterpoints=1, labelspacing=0.5, loc='lower left', handletextpad=1, frameon=False)
plt.gca().add_artist(legend2)  # Keep the first legend
# add a smooth grid
plt.grid(True, linestyle='--', alpha=0.5)

# Plot population legend in lower right corner
'''
for size in [min(population), np.mean(population), max(population)]:
    plt.scatter([], [], edgecolors='k', facecolors='none', s=size/max(population)*2000, 
                label=f'{size/1000000:.2f} M')

legend1 = plt.legend(title="Population", scatterpoints=1, labelspacing=0.5, loc='upper right', handletextpad=1.5)
plt.gca().add_artist(legend1)  # Keep the first legend
'''

plt.savefig("italy.png", dpi=600)
plt.show()

