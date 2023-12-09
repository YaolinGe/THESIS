""" Unit test for GRF
This module tests the GRF object.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-08-24
"""
from src.usr_func.interpolate_2d import interpolate_2d
from src.Config import Config
from unittest import TestCase
from src.GRF.GRF import GRF
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np


def plotf(grid, value, cmap: str = "BrBG", title: str = None, vmin: float = None, vmax: float = None):
    gx, gy, gv = interpolate_2d(grid[:, 0], grid[:, 1], 100, 100, value, "cubic")
    plt.scatter(gx, gy, c=gv, cmap=get_cmap(cmap, 10))
    plt.colorbar()
    plt.clim(vmin, vmax)
    plt.xlabel("East")
    plt.ylabel("North")
    plt.title(title)
    plt.xlim(np.amin(grid[:, 0]), np.amax(grid[:, 0]))
    plt.ylim(np.amin(grid[:, 1]), np.amax(grid[:, 1]))
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])


class TestGRF(TestCase):

    def setUp(self) -> None:
        self.c = Config()
        self.g = GRF()
        self.grid = self.g.grid
        self.cov = self.g.get_covariance_matrix()
        self.mu = self.g.get_mu()
        self.polygon_border = self.c.get_polygon_border()

    def test_show_grf(self) -> None:
        plt.figure(figsize=(10, 10))
        plotf(self.grid, self.mu)
        # plotf_vector(self.grid[:, 1], self.grid[:, 0], self.mu)
        plt.show()
        print("s")

    def test_assimilate(self):
        # c2: one
        print("S2")
        dataset = np.array([[3000, 1000, 10]])
        self.g.assimilate_data(dataset)
        plotf(self, v1=self.g.get_mu(), v2=np.sqrt(np.diag(self.g.get_covariance_matrix())),
              vmin1=10, vmax1=30, vmin2=0, vmax2=self.sigma, cbar1="salinity", cbar2="std", stepsize1=1.5, threshold1=27)

        # c3: multiple
        dataset = np.array([[2000, -1000, 15],
                            [1500, -1500, 10],
                            [1400, -1800, 25],
                            [2500, -1400, 20]])
        self.g.assimilate_data(dataset)
        plotf(self, v1=self.g.get_mu(), v2=np.sqrt(np.diag(self.g.get_covariance_matrix())),
              vmin1=10, vmax1=30, vmin2=0, vmax2=self.sigma, cbar1="salinity", cbar2="std", stepsize1=1.5, threshold1=27)
        print("End S2")

