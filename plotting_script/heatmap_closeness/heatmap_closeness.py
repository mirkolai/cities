import json
import matplotlib.pyplot as plt
import glob

files=glob.glob("json/heatmap_closeness-*.json")
for filename in files:

    data=json.load(open(filename))

    fig, ax = plt.subplots(figsize=(10, 10))


    scatter = ax.scatter(
        [ data["x"] for x in data if data["c"]],
        [ data["y"] for x in data if data["c"]],
        c=[ data["c"] for x in data if data["c"]],
        cmap="Blues",
        s=2,
        alpha=0.7,
        edgecolor=None,
        zorder=2,
    )

    ax.set_aspect('equal')

    ax.set_axis_off()

    plt.show()
    plt.close()