import unittest
import numpy as np
import matplotlib.pyplot as plt
from pathPlanningAlgorithms.PRM import PRM

class TestPRMPathPlanning(unittest.TestCase):
    def test_prm_path_planning(self):
        loc_start = [0, 0]
        loc_end = [.1, .8]
        polygon_border = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        polygon_obstacles = [np.array([[.1, .1], [.2, .1], [.2, .2], [.1, .2], [.1, .1]]),
                             np.array([[.7, .7], [.8, .7], [.8, .8], [.7, .8], [.7, .7]]),
                             np.array([[.3, .3], [.5, .3], [.5, .5], [.3, .5], [.3, .3]]),
                             np.array([[.6, .25], [.85, .25], [.85, .65], [.6, .65], [.6, .25]])]
        prm = PRM(polygon_border, polygon_obstacles)
        path = prm.get_path(loc_start, loc_end, num_nodes=1000, num_neighbours=10)
        plt.plot(polygon_border[:, 0], polygon_border[:, 1], 'r-.')
        for obs in polygon_obstacles:
            plt.plot(obs[:, 0], obs[:, 1], 'r-')
        for node in prm.nodes:
            if node.neighbours is not None:
                for i in range(len(node.neighbours)):
                    plt.plot([node.x, node.neighbours[i].x],
                             [node.y, node.neighbours[i].y], "g-", linewidth=0.1, alpha=0.5)
        plt.plot(loc_start[0], loc_start[1], 'bo')
        plt.plot(loc_end[0], loc_end[1], 'go')
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], 'b-')
        plt.show()
