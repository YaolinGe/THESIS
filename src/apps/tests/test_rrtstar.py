"""
Unittest for RRTStar.py
"""
import numpy as np
from unittest import TestCase
from pathPlanningAlgorithms.RRTStar import RRTStar


class RRTStarTestCase(TestCase):

    def setUp(self) -> None:
        polygon_border = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        polygon_obstacles = [np.array([[.1, .1], [.2, .1], [.2, .2], [.1, .2], [.1, .1]]),
                             np.array([[.7, .7], [.8, .7], [.8, .8], [.7, .8], [.7, .7]]),
                             np.array([[.3, .3], [.5, .3], [.5, .5], [.3, .5], [.3, .3]]),
                             np.array([[.6, .25], [.85, .25], [.85, .65], [.6, .65], [.6, .25]])]
        self.rrt_star = RRTStar(polygon_border=polygon_border, polygon_obstacles=polygon_obstacles,)

    def test_path_generation(self):
        loc_start = [0.01, 0.01]
        loc_end = [.3, .8]
        max_expansion_iteration = 800
        stepsize = .1
        home_radius = .08
        rrtstar_neighbour_radius = .12
        goal_sampling_rate = 0.01
        path = self.rrt_star.get_path(loc_start, loc_end, max_expansion_iteration, stepsize, goal_sampling_rate,
                                      home_radius=home_radius, rrtstar_neighbour_radius=rrtstar_neighbour_radius, animated=False)
        import matplotlib.pyplot as plt
        plt.plot(self.rrt_star.polygon_border[:, 0], self.rrt_star.polygon_border[:, 1], 'r-.')
        for obs in self.rrt_star.polygon_obstacles:
            plt.plot(obs[:, 0], obs[:, 1], 'r-')

        for node in self.rrt_star.nodes:
            if node.get_parent() is not None:
                loc = node.get_location()
                loc_p = node.get_parent().get_location()
                plt.plot([loc[0], loc_p[0]],
                         [loc[1], loc_p[1]], "-g")

        plt.plot(loc_start[0], loc_start[1], 'bo')
        plt.plot(loc_end[0], loc_end[1], 'go')
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], 'b-')
        plt.show()

    # def test_collision_avoidance(self) -> None:
        #     import matplotlib.pyplot as plt
        #
        #     loc1 = np.array([.09, .09])
        #     loc2 = np.array([.2, .2])
        #     isLegal = self.rrt_star.is_path_legal(loc1, loc2)
        #     self.assertFalse(isLegal)
        #
        #     plt.plot(self.rrt_star.polygon_border[:, 0], self.rrt_star.polygon_border[:, 1], 'r-.')
        #     for obs in self.rrt_star.polygon_obstacles:
        #         plt.plot(obs[:, 0], obs[:, 1], 'r-')
        #     plt.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], 'b-')
        #     print("Path is legal: ", isLegal)
        #     plt.show()
        #
        #     loc1 = np.array([.65, .72])
        #     loc2 = np.array([.85, .55])
        #     isLegal = self.rrt_star.is_path_legal(loc1, loc2)
        #     plt.figure()
        #     plt.plot(self.rrt_star.polygon_border[:, 0], self.rrt_star.polygon_border[:, 1], 'r-.')
        #     for obs in self.rrt_star.polygon_obstacles:
        #         plt.plot(obs[:, 0], obs[:, 1], 'r-')
        #     plt.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], 'b-')
        #     print("Path is legal: ", isLegal)
        #     plt.show()
