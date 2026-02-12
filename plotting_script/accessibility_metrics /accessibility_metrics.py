import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import matplotlib.ticker as mticker

data=json.load(open("json/accessibility_metrics.json"))
for place_name in data.keys():
    print(place_name)


    sns.set_style("white")

    plt.figure(figsize=(5, 15))

    metric_data = data[place_name]["proximity"]
    counts = np.array(metric_data['counts'])
    bins = np.array(metric_data['bins'])

    mean_val = metric_data['mean']
    min_val = metric_data['min']
    max_val = metric_data['max']
    std_val = metric_data['std']


    plt.subplot(4, 1, 1)
    print(len(counts),len(bins))
    plt.bar(bins[:-1], counts, width=np.diff(bins), color="green",
            edgecolor='black', align='edge', linewidth=0.5)

    x_ticks = [0,5,10,20,30,60]

    x_labels = ["0","5","10","20","30","60"]
    plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}%'))

    plt.xticks(x_ticks, x_labels)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0, 71)

    num_ticks = 6
    y_min, y_max = plt.ylim()

    yticks = np.linspace(y_min, y_max, num_ticks)

    plt.gca().set_yticks(yticks)

    textstr = '\n'.join((
        f'Avg: {mean_val:.2f}',
        f'Min: {min_val:.2f}',
        f'Max: {max_val:.2f}',
        f'Std: {std_val:.2f}'))

    props = dict(boxstyle='round', facecolor='white', alpha=0.8)

    plt.text(0.55, 0.95, textstr, transform=plt.gca().transAxes, fontsize=16,
             verticalalignment='top', horizontalalignment='left', bbox=props)

    ##########################################################
    metric_data = data[place_name]["density"]
    counts = np.array(metric_data['counts'])
    bins = np.array(metric_data['bins'])

    mean_val = metric_data['mean']
    min_val = metric_data['min']
    max_val = metric_data['max']
    std_val = metric_data['std']

    plt.subplot(4, 1, 2)
    plt.bar(bins[:-1], counts, width=np.diff(bins), color="blue",
            edgecolor='black', align='edge', linewidth=0.5)
    plt.xscale('log')


    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}%'))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0, 55)
    num_ticks = 6
    y_min, y_max = plt.ylim()

    yticks = np.linspace(y_min, y_max, num_ticks)

    plt.gca().set_yticks(yticks)
    textstr = '\n'.join((
        f'Avg: {mean_val:.2f}',
        f'Min: {min_val:.2f}',
        f'Max: {max_val:.2f}',
        f'Std: {std_val:.2f}'))

    props = dict(boxstyle='round', facecolor='white', alpha=0.8)

    plt.text(0.10, 0.95, textstr, transform=plt.gca().transAxes, fontsize=16,
             verticalalignment='top', horizontalalignment='left', bbox=props)



    ##########################################################
    metric_data = data[place_name]["entropy"]
    counts = np.array(metric_data['counts'])
    bins = np.array(metric_data['bins'])

    mean_val = metric_data['mean']
    min_val = metric_data['min']
    max_val = metric_data['max']
    std_val = metric_data['std']


    plt.subplot(4, 1, 3)
    plt.bar(bins[:-1], counts, width=np.diff(bins), color="red",
            edgecolor='black', align='edge', linewidth=0.5)

    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}%'))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0, 69)
    num_ticks = 6
    y_min, y_max = plt.ylim()

    yticks = np.linspace(y_min, y_max, num_ticks)
    plt.xlim(0, 1)

    plt.gca().set_yticks(yticks)
    textstr = '\n'.join((
        f'Avg: {mean_val:.2f}',
        f'Min: {min_val:.2f}',
        f'Max: {max_val:.2f}',
        f'Std: {std_val:.2f}'))

    props = dict(boxstyle='round', facecolor='white', alpha=0.8)

    plt.text(0.10, 0.95, textstr, transform=plt.gca().transAxes, fontsize=16,
             verticalalignment='top', horizontalalignment='left', bbox=props)


    ##########################################################
    metric_data = data[place_name]["PoI-Accessibility"]
    counts = np.array(metric_data['counts'])
    bins = np.array(metric_data['bins'])

    mean_val = metric_data['mean']
    min_val = metric_data['min']
    max_val = metric_data['max']
    std_val = metric_data['std']


    plt.subplot(4, 1, 4)
    plt.bar(bins[:-1], counts, width=np.diff(bins), color="yellow",
            edgecolor='black', align='edge', linewidth=0.5)

    plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}%'))
    plt.xticks(fontsize=15)
    plt.xlim(0, 1)  #
    plt.yticks(fontsize=15)
    plt.ylim(0, 68)
    num_ticks = 6
    y_min, y_max = plt.ylim()

    yticks = np.linspace(y_min, y_max, num_ticks)

    plt.gca().set_yticks(yticks)
    textstr = '\n'.join((
        f'Avg: {mean_val:.2f}',
        f'Min: {min_val:.2f}',
        f'Max: {max_val:.2f}',
        f'Std: {std_val:.2f}'))

    props = dict(boxstyle='round', facecolor='white', alpha=0.8)

    plt.text(0.10, 0.95, textstr, transform=plt.gca().transAxes, fontsize=16,
             verticalalignment='top', horizontalalignment='left', bbox=props)




    plt.tight_layout()
    plt.show()
    plt.close()



