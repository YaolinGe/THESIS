""" Unittest for GRF.py """
from unittest import TestCase
from GRF import GRF
import numpy as np
import os
import matplotlib.pyplot as plt


class TestGRF(TestCase):

    def setUp(self) -> None:
        self.polygon_border = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.polygon_obstacle = np.empty((0, 2))
        self.grid_size = .05
        self.lateral_range = .6
        self.sigma = .2
        self.nugget = .01
        self.threshold = .5
        self.grf = GRF(self.polygon_border, self.polygon_obstacle, self.grid_size, self.lateral_range, self.sigma, self.nugget, self.threshold)

    def test_get_random_realization(self) -> None:
        self.grf.get_random_realization()

    def test_get_excursion_set(self) -> None:
        es = self.grf.get_excursion_set()
        plt.scatter(self.grf.grid[:, 0], self.grf.grid[:, 1], c=es, cmap='Paired', vmin=0, vmax=1)
        plt.colorbar()
        plt.show()

    def test_get_excursion_probability(self) -> None:
        ep = self.grf.get_excursion_probability()
        plt.scatter(self.grf.grid[:, 0], self.grf.grid[:, 1], c=ep, cmap='GnBu', vmin=0, vmax=1)
        plt.colorbar()
        plt.show()
