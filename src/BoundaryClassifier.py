"""
This script is used to produce figures used in the thesis for the Plan section.

Author: Yaolin Ge
Email: gayaolin@gmail.com
Date: 2023-08-09

"""
from WGS import WGS
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
from SINMOD import SINMOD
import pandas as pd


def plotf_vector(xplot, yplot, values, title=None, alpha=None, cmap=get_cmap("BrBG", 10),
                 cbar_title='test', colorbar=True, vmin=None, vmax=None, ticks=None,
                 stepsize=None, threshold=None, polygon_border=None,
                 polygon_obstacle=None, xlabel=None, ylabel=None):
    """ Note for triangulation:
    - Maybe sometimes it cannot triangulate based on one axis, but changing to another axis might work.
    - So then the final output needs to be carefully treated so that it has the correct visualisation.
    - Also note, the floating point number can cause issues as well.
    """
    """ To show threshold as a red line, then vmin, vmax, stepsize, threshold needs to have values. """
    triangulated = tri.Triangulation(xplot, yplot)
    x_triangulated = xplot[triangulated.triangles].mean(axis=1)
    y_triangulated = yplot[triangulated.triangles].mean(axis=1)

    ind_mask = []
    for i in range(len(x_triangulated)):
        ind_mask.append(is_masked(y_triangulated[i], x_triangulated[i]))

    triangulated.set_mask(ind_mask)
    refiner = tri.UniformTriRefiner(triangulated)
    triangulated_refined, value_refined = refiner.refine_field(values.flatten(), subdiv=3)

    """ extract new x and y, refined ones. """
    # xre_plot = triangulated_refined.x
    # yre_plot = triangulated_refined.y
    ax = plt.gca()
    # ax.triplot(triangulated, lw=0.5, color='white')
    if np.any([vmin, vmax]):
        levels = np.arange(vmin, vmax, stepsize)
    else:
        levels = None
    if np.any(levels):
        linewidths = np.ones_like(levels) * .3
        colors = len(levels) * ['black']
        if threshold:
            dist = np.abs(threshold - levels)
            ind = np.where(dist == np.amin(dist))[0]
            linewidths[ind] = 10
            colors[ind[0]] = 'red'
        contourplot = ax.tricontourf(triangulated_refined, value_refined, levels=levels, cmap=cmap, alpha=alpha)
        ax.tricontour(triangulated_refined, value_refined, levels=levels, linewidths=linewidths, colors=colors,
                      alpha=alpha)
        # contourplot = ax.tricontourf(yre_plot, xre_plot, value_refined, levels=levels, cmap=cmap, alpha=alpha)
        # ax.tricontour(yre_plot, xre_plot, value_refined, levels=levels, linewidths=linewidths, colors=colors,
        #               alpha=alpha)
    else:
        contourplot = ax.tricontourf(triangulated_refined, value_refined, cmap=cmap, alpha=alpha)
        ax.tricontour(triangulated_refined, value_refined, vmin=vmin, vmax=vmax, alpha=alpha)
        # contourplot = ax.tricontourf(yre_plot, xre_plot, value_refined, cmap=cmap, alpha=alpha)
        # ax.tricontour(yre_plot, xre_plot, value_refined, vmin=vmin, vmax=vmax, alpha=alpha)

    if colorbar:
        cbar = plt.colorbar(contourplot, ax=ax, ticks=ticks)
        cbar.ax.set_title(cbar_title)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if np.any(polygon_border):
        plt.plot(polygon_border[:, 1], polygon_border[:, 0], 'k-.', lw=2)

    if np.any(polygon_obstacle):
        plt.plot(polygon_obstacle[:, 1], polygon_obstacle[:, 0], 'k-.', lw=2)
    return ax, value_refined


def is_masked(xgrid, ygrid) -> bool:
    """
    :param xgrid:
    :param ygrid:
    :return:
    """
    loc = np.array([xgrid, ygrid])
    masked = False
    if field.obstacle_contains(loc) or not field.border_contains(loc):
        masked = True
    return masked


polygon_border = pd.read_csv("csv/polygon_border.csv").to_numpy()
data_sinmod = pd.read_csv("csv/sinmod_surface_2022.05.04.csv").to_numpy()

lat_sinmod, lon_sinmod = WGS.xy2latlon(data_sinmod[:, 0], data_sinmod[:, 1])
sal_sinmod = data_sinmod[:, -1]

plt.figure(figsize=(10, 10))
plt.scatter(lon_sinmod, lat_sinmod, c=sal_sinmod, cmap=get_cmap("BrBG", 10), vmin=10, vmax=33)
plt.plot(polygon_border[:, 1], polygon_border[:, 0], "r-.")
plt.title("SINMOD salinity at surface at " + "2022.05.04")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.colorbar()
plt.show()



