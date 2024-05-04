"""
Unittest for AStar.py
"""
import numpy as np
from unittest import TestCase
from pathPlanningAlgorithms.AStar import AStar


class RRTStarTestCase(TestCase):

    def setUp(self) -> None:
        polygon_border = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        polygon_obstacles = [np.array([[.1, .1], [.2, .1], [.2, .2], [.1, .2], [.1, .1]]),
                             np.array([[.7, .7], [.8, .7], [.8, .8], [.7, .8], [.7, .7]]),
                             np.array([[.3, .3], [.5, .3], [.5, .5], [.3, .5], [.3, .3]]),
                             np.array([[.6, .25], [.85, .25], [.85, .65], [.6, .65], [.6, .25]])]
        self.astar = AStar(polygon_border=polygon_border, polygon_obstacles=polygon_obstacles,)

    def test_path_generation(self):
        loc_start = [0.01, 0.01]
        loc_end = [.95, .675]
        max_iter = 2000
        stepsize = .05
        distance_tolerance_target = .075
        distance_tolerance = .02
        path = self.astar.search_path(loc_start, loc_end, max_iter, stepsize, distance_tolerance_target, distance_tolerance)
        path = np.array(path)
        import matplotlib.pyplot as plt
        plt.plot(self.astar.polygon_border[:, 0], self.astar.polygon_border[:, 1], 'r-.')
        for obs in self.astar.polygon_obstacles:
            plt.plot(obs[:, 0], obs[:, 1], 'r-')
        plt.plot(loc_start[0], loc_start[1], 'bo')
        plt.plot(loc_end[0], loc_end[1], 'go')
        plt.plot(path[:, 0], path[:, 1], 'b-')
        plt.show()
                