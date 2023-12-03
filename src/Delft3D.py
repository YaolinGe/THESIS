"""
Script used to generate the prior data for Delft3D model to be used for GIS visualization.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-11-27
"""
import os
import numpy as np
import h5py
import pandas as pd
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
from time import time


# s0, load data
datapath = "/Users/yaolin/CodeSpace/OASIS/Server/raw/March/"
files = os.listdir(datapath)
files.sort()
file = files[0]
df = h5py.File(datapath + file, 'r')

# s1: get lat,lon
t1 = time()
data = df.get('data')
lon = np.array(data.get("X"))
lat = np.array(data.get("Y"))
depth = np.array(data.get("Z"))
salinity = np.array(data.get("Val"))
timestamp = np.array(data.get("Time"))
t2 = time()
print("Loading time takes: ", t2 - t1)

#%%
# s2, get surface average salinity
sal_surface = np.mean(salinity[0, :, :], axis=-1)

#%%
lat = lat[0, :, :, 0]
lon = lon[0, :, :, 0]

#%%
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm

# Use a non-interactive backend
plt.switch_backend('Agg')

# Create the colormap and normalization once
cmap = get_cmap("BrBG", 10)
norm = plt.Normalize(vmin=10, vmax=36)

# Create a single figure and axes

for i in tqdm(range(220, salinity.shape[-1])):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create the scatter plot
    sc = ax.scatter(lon, lat, c=salinity[0, :, :, i], cmap=cmap, norm=norm)

    # Add colorbar
    fig.colorbar(sc, ax=ax)

    # Set title
    ax.set_title("Time: " + str(i))

    # Save the figure
    fig.savefig(os.path.join("/Users/yaolin/Downloads/fig/", f"P_{i:03d}.png"))

    # Close the plot to free memory
    plt.close(fig)

# Close the figure finally

# for i in tqdm(range(220, salinity.shape[-1])):
#     fig = plt.figure(figsize=(10, 10))
#     plt.scatter(lon, lat, c=salinity[0, :, :, i], cmap=get_cmap("BrBG", 10), vmin=10, vmax=36)
#     plt.colorbar()
#     plt.title("Time: " + str(i))
#     plt.savefig("/Users/yaolin/Downloads/fig/P_{:03d}.png".format(i))
#     plt.close("all")


plt.switch_backend('TkAgg')  # or 'Qt5Agg', 'MacOSX', 'nbAgg', etc., depending on your environment

#%%
# save data to csv for time step 155
df = pd.DataFrame()
df["lon"] = lon.flatten()
df["lat"] = lat.flatten()
df["salinity"] = salinity[0, :, :, 155].flatten()
df.to_csv("/Users/yaolin/Downloads/fig/P_155.csv", index=False)

