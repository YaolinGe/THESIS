"""
RRTStar object produces the possible tree generation in the constrained field.
It employs RRT as the building block, and the cost associated with each tree branch is used to
determine the final tree discretization.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2024-05-03
"""
import numpy as np
import math
import streamlit as st
import plotly.graph_objects as go
from time import time
from shapely.geometry import Polygon, Point, LineString


class TreeNode:

    __x = .0
    __y = .0
    __cost = .0
    __parent = None

    def __init__(self, loc: np.ndarray, cost=0, parent=None) -> None:
        self.__x, self.__y = loc
        self.__cost = cost
        self.__parent = parent

    def set_location(self, loc: np.ndarray) -> None:
        """ Set location for the new tree node. """
        self.__x, self.__y = loc

    def set_cost(self, value: float) -> None:
        """ Set cost associated with tree node. """
        self.__cost = value

    def set_parent(self, parent: 'TreeNode') -> None:
        """ Set parent of the current tree node. """
        self.__parent = parent

    def get_location(self) -> np.ndarray:
        """ Return the location associated with the tree node. """
        return np.array([self.__x, self.__y])

    def get_cost(self) -> float:
        """ Get cost associated with the tree node. """
        return self.__cost

    def get_parent(self):
        """ Return the parent node of the tree node. """
        return self.__parent

    @staticmethod
    def get_distance_between_nodes(n1: 'TreeNode', n2: 'TreeNode') -> float:
        dist = np.sqrt((n1.__x - n2.__x)**2 +
                       (n1.__y - n2.__y)**2)
        return dist


class RRTStar:
    """ RRTStar algorithm """
    def __init__(self, polygon_border: np.ndarray, polygon_obstacles: list) -> None:
        self.polygon_border = polygon_border
        self.polygon_obstacles = polygon_obstacles
        self.polygon_border_shapely = Polygon(polygon_border)
        self.polygon_obstacle_shapely = [Polygon(o) for o in polygon_obstacles]
        self.line_border_shapely = LineString(polygon_border)
        self.line_obstacle_shapely = [LineString(o) for o in polygon_obstacles]

        self.xlim = [min(polygon_border[:, 0]), max(polygon_border[:, 0])]
        self.ylim = [min(polygon_border[:, 1]), max(polygon_border[:, 1])]

    def get_path(self, loc_start: np.ndarray, loc_target: np.ndarray, max_expansion_iteration: int=1700,
                 stepsize: float=.1, goal_sampling_rate: float=.01, home_radius: float=.08,
                 rrtstar_neighbour_radius: float=.12, animated: bool=False) -> np.ndarray:
        """
        Get the next waypoint according to RRT* path planning philosophy.
        :param loc_start: current location np.array([x, y])
        :param loc_target: minimum cost location, np.array([x, y])
        :param cost_valley: cost valley contains the cost field.
        :return next waypoint: np.array([x, y])
        """
        self.animated = animated
        self.loc_start = loc_start
        self.loc_target = loc_target
        self.loc_new = loc_target

        # tree
        self.nodes = []  # all nodes in the tree.
        self.trajectory = np.empty([0, 2])  # to save trajectory.
        self.cost_trajectory = .0  # cost along the trajectory.
        self.distance_trajectory = .0  # distance along the trajectory.
        self.goal_sampling_rate = goal_sampling_rate
        self.max_expansion_iteration = max_expansion_iteration
        self.stepsize = stepsize
        self.home_radius = home_radius
        self.rrtstar_neighbour_radius = rrtstar_neighbour_radius
        # self.home_radius = self.stepsize * .8
        # self.rrtstar_neighbour_radius = self.stepsize * 1.12

        # nodes # s1: set starting location and target location in rrt*.
        self.starting_node = TreeNode(self.loc_start)
        self.target_node = TreeNode(self.loc_target)
        self.nearest_node = None
        self.new_node = None
        self.neighbour_nodes = []

        t_start = time()
        # s2: expand the trees.
        self.expand_trees_and_find_shortest_trajectory()

        t_end = time()
        print("RRT* time: ", t_end - t_start, "s")
        return self.trajectory

    def expand_trees_and_find_shortest_trajectory(self):
        # start by appending the starting node to the nodes list.
        self.nodes.append(self.starting_node)
        if self.animated:
            fig = go.Figure()
            fig.update_layout(
                width=500, 
                height=700,
                showlegend=False,
                )
            fig.add_trace(go.Scatter(x=self.polygon_border[:, 0], y=self.polygon_border[:, 1], mode='lines', line=dict(color='red', dash='dash')))
            for obs in self.polygon_obstacles:
                fig.add_trace(go.Scatter(x=obs[:, 0], y=obs[:, 1], mode='lines', line=dict(color='red')))
            loc_nodes = np.array([node.get_location() for node in self.nodes])
            fig.add_trace(go.Scatter(x=loc_nodes[:, 0], y=loc_nodes[:, 1], mode='markers', marker=dict(size=7, color='white')))
            fig.add_trace(go.Scatter(x=[self.loc_start[0]], y=[self.loc_start[1]], mode='markers', marker=dict(size=20, color='green')))
            fig.add_trace(go.Scatter(x=[self.loc_target[0]], y=[self.loc_target[1]], mode='markers', marker=dict(size=20, color='blue')))
            progress_bar = st.progress(0)
            self.chart_placeholder = st.empty()

        for i in range(self.max_expansion_iteration):
            # s1: get new location.
            if np.random.uniform() <= self.goal_sampling_rate:
                self.loc_new = self.loc_target
            else:
                self.loc_new = self.get_random_location()

            # s2: get nearest node to the current location.
            self.get_nearest_node()

            # s3: steer new location to get the nearest tree node to this new location.
            if TreeNode.get_distance_between_nodes(self.nearest_node, self.new_node) > self.stepsize:
                xn, yn = self.nearest_node.get_location()
                angle = math.atan2(self.loc_new[1] - yn,
                                   self.loc_new[0] - xn)
                x = xn + self.stepsize * np.cos(angle)
                y = yn + self.stepsize * np.sin(angle)
                loc = np.array([x, y])
                self.new_node = TreeNode(loc, parent=self.nearest_node)

            # s4: check if it is colliding.
            if not self.is_location_legal(self.new_node.get_location()):
                continue

            # s5: rewire trees in the neighbourhood.
            self.rewire_trees()

            # s6: check path possibility.
            if not self.is_path_legal(self.nearest_node.get_location(),
                                      self.new_node.get_location()):
                continue

            # s7: check connection to the goal node.
            if self.isarrived():
                self.target_node.set_parent(self.new_node)
                self.target_node.set_cost(self.get_cost_between_nodes(self.target_node, self.new_node))
            else:
                self.nodes.append(self.new_node)

            ### Plotting section for animation
            if self.animated:
                # neighbours = []
                # for node in self.nodes:
                #     if node.get_parent() is not None:
                #         loc = node.get_location()
                #         loc_p = node.get_parent().get_location()
                #         neighbours.append([loc[0], loc_p[0], loc[1], loc_p[1]])
                # neighbours = np.array(neighbours)
                # for n in neighbours:
                #     fig.add_trace(go.Scatter(x=[n[0], n[1]], y=[n[2], n[3]], mode='lines', line=dict(color='white', width=1)))

                fig.add_trace(go.Scatter(x=[self.new_node.get_location()[0], self.new_node.get_parent().get_location()[0]], 
                                         y=[self.new_node.get_location()[1], self.new_node.get_parent().get_location()[1]], 
                                         mode='lines', line=dict(color='white', width=1)))

                self.chart_placeholder.plotly_chart(fig, use_container_width=True)
                progress_bar.progress((i + 1) / self.max_expansion_iteration)
        
        if self.animated:
            progress_bar.empty()

        # s8: get shortest trajectory.
        self.get_shortest_trajectory()
            
        if self.animated: 
            fig.add_trace(go.Scatter(x=self.trajectory[:, 0], y=self.trajectory[:, 1], mode='lines', line=dict(color='yellow', width=5)))
            self.chart_placeholder.plotly_chart(fig, use_container_width=True)

    def get_random_location(self) -> np.ndarray:
        """ Returns a legal random location within the field. """
        x = np.random.uniform(self.xlim[0], self.xlim[1])
        y = np.random.uniform(self.ylim[0], self.ylim[1])
        loc = np.array([x, y])
        while not self.is_location_legal(loc):
            x = np.random.uniform(self.xlim[0], self.xlim[1])
            y = np.random.uniform(self.ylim[0], self.ylim[1])
            loc = np.array([x, y])
        return loc

    def get_nearest_node(self) -> None:
        """ Return nearest node in the tree graph, only use distance. """
        dist = []
        self.new_node = TreeNode(self.loc_new)
        for node in self.nodes:
            dist.append(TreeNode.get_distance_between_nodes(node, self.new_node))
        self.nearest_node = self.nodes[dist.index(min(dist))]
        self.new_node.set_parent(self.nearest_node)

    def rewire_trees(self):
        # s1: find cheapest node.
        self.get_neighbour_nodes()
        for node in self.neighbour_nodes:
            if (self.get_cost_between_nodes(node, self.new_node) <
                    self.get_cost_between_nodes(self.nearest_node, self.new_node)):
                self.nearest_node = node

            self.new_node.set_parent(self.nearest_node)
            self.new_node.set_cost(self.get_cost_between_nodes(self.nearest_node, self.new_node))

        # s2: update other nodes.
        for node in self.neighbour_nodes:
            cost_current_neighbour = self.get_cost_between_nodes(self.new_node, node)
            if cost_current_neighbour < node.get_cost():
                node.set_cost(cost_current_neighbour)
                node.set_parent(self.new_node)

    def get_neighbour_nodes(self):
        distance_between_nodes = []
        for node in self.nodes:
            distance_between_nodes.append(TreeNode.get_distance_between_nodes(node, self.new_node))
        ind_neighbours = np.where(np.array(distance_between_nodes) <= self.rrtstar_neighbour_radius)[0]
        self.neighbour_nodes = []
        for idx in ind_neighbours:
            self.neighbour_nodes.append(self.nodes[idx])

    def get_cost_between_nodes(self, n1: 'TreeNode', n2: 'TreeNode') -> float:
        """ Get cost between nodes. """
        cost_distance = TreeNode.get_distance_between_nodes(n1, n2) / self.stepsize
        if self.is_path_legal(n1.get_location(), n2.get_location()):
            cost_collision = 0
        else:
            cost_collision = 100
        cost = n1.get_cost() + cost_distance + cost_collision
        return cost

    def isarrived(self) -> bool:
        dist = TreeNode.get_distance_between_nodes(self.new_node, self.target_node)
        if dist < self.home_radius:
            return True
        else:
            return False

    def get_shortest_trajectory(self):
        wp_old = self.target_node.get_location().reshape(1, -1)
        self.trajectory = np.empty([0, 2])
        self.trajectory = np.append(self.trajectory, wp_old, axis=0)
        self.cost_trajectory = self.target_node.get_cost()

        pointer_node = self.target_node
        cnt = 0
        while pointer_node.get_parent() is not None:
            cnt += 1
            node = pointer_node.get_parent()
            wp_new = pointer_node.get_location().reshape(1, -1)
            self.trajectory = np.append(self.trajectory, wp_new, axis=0)
            self.distance_trajectory += np.sqrt((wp_new[0, 0] - wp_old[0, 0]) ** 2 +
                                                (wp_new[0, 1] - wp_new[0, 1]) ** 2)

            pointer_node = node
            wp_old = wp_new

            if cnt > self.max_expansion_iteration:
                break

        wp_new = self.starting_node.get_location().reshape(1, -1)
        self.trajectory = np.append(self.trajectory, wp_new, axis=0)
        self.cost_trajectory += self.starting_node.get_cost()
        self.distance_trajectory += np.sqrt((wp_new[0, 0] - wp_old[0, 0]) ** 2 +
                                            (wp_new[0, 1] - wp_old[0, 1]) ** 2)

    def get_tree_nodes(self) -> list:
        """ Return all the tree nodes. """
        return self.nodes

    def get_trajectory(self) -> np.ndarray:
        """ Return the trajectory from the starting location to the target location. """
        return self.trajectory

    def get_cost_along_trajectory(self) -> float:
        """ Get the cost along the desired trajectory. """
        return self.cost_trajectory

    def get_distance_along_trajectory(self) -> float:
        return self.distance_trajectory

    def is_location_legal(self, loc: np.ndarray) -> bool:
        x, y = loc
        point = Point(x, y)
        islegal = True
        for polygon_obstacle in self.polygon_obstacle_shapely:
            if polygon_obstacle.contains(point):
                islegal = False
                break
        return islegal

    def is_path_legal(self, loc1: np.ndarray, loc2: np.ndarray) -> bool:
        x1, y1 = loc1
        x2, y2 = loc2
        line = LineString([(x1, y1), (x2, y2)])
        islegal = True
        c1 = self.line_border_shapely.intersects(line)  # TODO: tricky to detect, since cannot have points on border.
        for polygon_obstacle in self.polygon_obstacle_shapely:
            c2 = polygon_obstacle.intersects(line)
            if c2:
                islegal = False
                break
        if c1:
            islegal = False
        return islegal

    def set_goal_sampling_rate(self, value: float) -> None:
        """ Set the goal sampling rate. """
        self.goal_sampling_rate = value

    def set_stepsize(self, value: float) -> None:
        """ Set the step size of the trees. """
        self.stepsize = value

    def set_max_expansion_iteraions(self, value: int) -> None:
        """ Set the maximum expansion itersions. """
        self.max_expansion_iteration = value

    def set_rrtstar_neighbour_radius(self, value: float) -> None:
        """ Set the neighbour radius for tree searching. """
        self.rrtstar_neighbour_radius = value

    def set_home_radius(self, value: float) -> None:
        """ Set the home radius for path convergence. """
        self.home_radius = value

    def get_goal_sampling_rate(self) -> float:
        """ Get the goal sampling rate. """
        return self.goal_sampling_rate

    def get_stepsize(self) -> float:
        """ Get the step size of the trees. """
        return self.stepsize

    def get_max_expansion_iteraions(self) -> int:
        """ Get the maximum expansion itersions. """
        return self.max_expansion_iteration

    def get_rrtstar_neighbour_radius(self) -> float:
        """ Get the neighbour radius for tree searching. """
        return self.rrtstar_neighbour_radius

    def get_home_radius(self) -> float:
        """ Get the home radius for path convergence. """
        return self.home_radius


if __name__ == "__main__":
    t = RRTStar()



