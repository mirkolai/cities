import utils
import numpy as np
import matplotlib.pyplot as plt
import json
import glob


def round_to_significant(number):
    number = int(number)
    return round(number,-1) if round(number,-1) >=1 else 1


def scale_value(value, min_val, max_val, min_size, max_size):
    return min_size + (max_size - min_size) * (value - min_val) / (max_val - min_val)


files=glob.glob("json/scatter_AvsC-*.json")
for filename in files:
    data=json.load(open(filename))
    categories = data["meta"]["categories"]
    min_size = data["meta"]["min_size"]
    max_size = data["meta"]["max_size"]
    pop_min=data["meta"]["pop_min"]
    pop_max=data["meta"]["pop_max"]

    fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.1, hspace=0.1)

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    x = np.array([p["x"] for p in data["points"]])
    y = np.array([p["y"] for p in data["points"]])
    s = np.array([p["population"] for p in data["points"]])
    neighborhoods = np.array([p["neighborhood"] for p in data["points"]])

    bins_x = np.histogram_bin_edges(x, bins=30)
    bins_y = np.histogram_bin_edges(y, bins=30)

    population_sum_x = [
        s[(x >= bins_x[i]) & (x < bins_x[i + 1])].sum()
        for i in range(len(bins_x) - 1)
    ]

    population_sum_y = [
        s[(y >= bins_y[i]) & (y < bins_y[i + 1])].sum()
        for i in range(len(bins_y) - 1)
    ]

    ax.set_xlabel('Normalized Closeness', fontsize=14)
    ax.set_ylabel('PoI-Accessibility', fontsize=14)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    for category in categories:
        mask = neighborhoods == category
        subset_x = x[mask]
        subset_y = y[mask]
        subset_s = s[mask]
        if subset_s.sum()<1000:
            continue
        if len(subset_s)==0:
            continue

        def scale_value(value, min_val, max_val, min_size, max_size):
            return min_size + (max_size - min_size) * (value - min_val) / (max_val - min_val)

        ax.scatter(
            subset_x,
            subset_y,
            color=utils.cluster_colors[int(category)%len(utils.cluster_colors)],
            label=category,
            s=[min_size + (max_size - min_size) * (v - pop_min) / (pop_max - pop_min) for v in subset_s],
            edgecolor='k',
            alpha=0.5,
            zorder=int((1/subset_s.sum())*100000)
        )

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

    unique_sizes = [round_to_significant(pop_min),round_to_significant((pop_min+pop_max)/2),round_to_significant(pop_max)]
    handles = [plt.scatter([], [], s=scale_value(size, pop_min, pop_max, min_size, max_size), edgecolor='k',
                           color='white', alpha=0.3) for size in unique_sizes]
    labels = [f'{int(size)}' for size in unique_sizes]
    legend = ax.legend(handles, labels, title="Population", loc="upper left", frameon=False, fontsize=10, title_fontsize=12)

    ax_histx.bar(
        bins_x[:-1],
        population_sum_x,
        width=np.diff(bins_x),
        color='lightgrey',
        edgecolor='black',
        align='edge'
    )
    ax_histy.barh(
        bins_y[:-1],
        population_sum_y,
        height=np.diff(bins_y),
        color='lightgrey',
        edgecolor='black',
        align='edge'
    )

    ax_histx.set_ylabel('Population', labelpad=2, fontsize=14)
    ax_histy.set_xlabel('Population', labelpad=2, fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.close()
