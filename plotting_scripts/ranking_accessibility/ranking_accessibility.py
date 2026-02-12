import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import utils

data = json.load(open("json/ranking_accessibility.json"))
sorted_cities = sorted(data.items(), key=lambda x: x[1]["mean"],reverse=True)
sorted_city_names = [city for city, _ in sorted_cities]
print(sorted_city_names)

bxp_data = []
for city, s in sorted_cities:
    IQR = s["q3"] - s["q1"]
    whis=1.5
    whisker_low = max(s["min"], s["q1"] - whis * IQR)
    whisker_hight = min(s["max"], s["q3"] + whis * IQR)
    bxp_data.append({
        "mean": s["mean"],
        "med": s["median"],
        "q1": s["q1"],
        "q3": s["q3"],
        "whislo": whisker_low,
        "whishi": whisker_hight,
        "fliers": []
    })

fig, ax = plt.subplots(figsize=(12, 6))
boxplot = ax.bxp(
    bxp_data,
    showmeans=True,
    meanline=True,
    showfliers=False,
    patch_artist=True,
    widths=1
)
for median in boxplot['medians']:
    median.set_color('white')
    median.set_linestyle(' ')
    median.set_linewidth(0)
for median in boxplot['means']:
    median.set_color('white')
    median.set_linestyle(' ')
    median.set_linewidth(0)


print(boxplot.keys())
for i,patch in enumerate(boxplot['boxes']):

    patch.set_facecolor(utils.continent_to_color[utils.city_to_continent[sorted_city_names[i]]])


medians = [s["median"] for _, s in sorted_cities]
means = [s["mean"] for _, s in sorted_cities]

plt.scatter(
    range(1, len(sorted_city_names) + 1),
    medians,
    color='black',
    marker='.',
    s=80,
    label='Median',
    zorder=3
)

plt.scatter(
    range(1, len(sorted_city_names) + 1),
    means,
    color='black',
    marker='+',
    s=80,
    label='Mean',
    zorder=3
)
color_handles = [
    plt.scatter([], [], s=100, color=color, edgecolor='k')
    for color in utils.continent_to_color.values()
]
color_labels = list(utils.continent_to_color.keys())

mean_handle = mlines.Line2D([], [], color='black', marker='+',
                           linestyle='None', markersize=8, label='Mean')
median_handle = mlines.Line2D([], [], color='black', marker='.',
                             linestyle='None', markersize=8, label='Median')

color_handles.extend([median_handle, mean_handle])
color_labels.extend(['Median', 'Mean'])

legend2 = plt.legend(
    color_handles,
    color_labels,
    title="", loc="upper right",
    frameon=False,
    fontsize=10,title_fontsize=12,
    bbox_to_anchor=(0.455, 0.18), ncol=3
)

plt.xlim(0.5, len(sorted_city_names) + 0.5)
plt.xticks(range(1, len(sorted_city_names) + 1), sorted_city_names, rotation=90)
plt.ylabel("PoI-Accessibility Distribution")
plt.xlabel("Cities")
plt.title("")

plt.show()

