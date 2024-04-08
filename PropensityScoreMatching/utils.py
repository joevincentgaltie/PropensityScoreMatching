import numpy as np 
import pandas as pd 
from cycler import cycler
import matplotlib.pyplot as plt



# Boostraping methods
def bootstrap_mean(data  : pd.Series) -> np.array:

    return np.mean(np.random.choice(data, len(data)))

def bootstrap_mean_confidence_interval(data : pd.Series, n_replicates  : int = 10000, percentiles : list = [2.5, 97.5]) -> np.array:

    replicates=np.empty(n_replicates)

    for i in range(n_replicates):
        replicates[i]=bootstrap_mean(data)

    return np.percentile(replicates,percentiles)

# Plot parameters 
def set_graph_style():
    """Sets the default style for all graphs."""



    # Figure size
    plt.rcParams['figure.figsize'] = (15, 7)

    # Line plot styles
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 8

    # Axis labels and ticks
    plt.rcParams['font.family'] = 'Cambria'
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16

    # Legend
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['legend.title_fontsize'] = 16
    plt.rcParams['legend.framealpha'] = 0

    plt.rcParams['legend.loc'] = 'upper center'
    #plt.rcParams['legend.bbox_to_anchor'] = (0.5, -0.05)
    #plt.rcParams['legend.ncol'] = 3


    # Remove top and right spines
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True


    # Set custom colormap
    plt.rcParams['axes.prop_cycle'] = cycler('color', ['#096c45', '#d69a00', '#9f0025', "#e17d18"])