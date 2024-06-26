# """
# PRM conducts the path planning using probabilistic road map philosophy. It selects the minimum cost path between
# the starting location and the end location.
# """
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import copy
# from shapely.geometry import Point, Polygon, LineString


# class Node:

#     def __init__(self, loc=None, cost=None):
#         self.x, self.y = loc
#         self.cost = cost
#         self.parent = None
#         self.neighbours = []

# class PRM:

#     def __init__(self, border, obstacles):
#         self.nodes = []
#         self.plg_border = border
#         self.plg_obstacles = obstacles
#         self.plg_border_shapely = Polygon(self.plg_border)
#         self.plg_obstacle_shapely = [Polygon(plg_obstacle) for plg_obstacle in self.plg_obstacles]
#         self.xlim = [np.amin(border[:, 0]), np.amax(border[:, 0])]
#         self.ylim = [np.amin(border[:, 1]), np.amax(border[:, 1])]
#         self.path = []

#     def get_path(self, loc_start, loc_end, num_nodes=1000, num_neighbours=10):
#         self.nodes = []
#         self.path = []
#         self.loc_start = loc_start
#         self.loc_end = loc_end
#         self.num_nodes = num_nodes
#         self.num_neighbours = num_neighbours
#         self.get_road_map()
#         self.get_shortest_path_using_dijkstra()
#         if self.path:
#             return self.path
#         else:
#             return None

#     def get_road_map(self):
#         # s1: initialise nodes
#         self.starting_node = Node(self.loc_start)
#         self.ending_node = Node(self.loc_end)

#         # s2: get random locations
#         self.nodes.append(self.starting_node)
#         counter_nodes = 0
#         while counter_nodes < self.num_nodes:
#             new_location = self.get_new_location()
#             if not self.inRedZone(new_location):
#                 self.nodes.append(Node(new_location))
#                 counter_nodes += 1
#         self.nodes.append(self.ending_node)

#         # s3: get road maps
#         for i in range(len(self.nodes)):
#             dist = []
#             node_now = self.nodes[i]
#             for j in range(len(self.nodes)):
#                 node_next = self.nodes[j]
#                 dist.append(PRM.get_distance_between_nodes(node_now, node_next))
#             ind_sort = np.argsort(dist)
#             # print(ind_sort[:self.config.num_neighbours])
#             for k in range(self.num_neighbours):
#                 node_neighbour = self.nodes[ind_sort[:self.num_neighbours][k]]
#                 if not self.iscollided(node_now, node_neighbour):
#                     node_now.neighbours.append(node_neighbour)

#     # @staticmethod
#     def get_new_location(self):
#         while True:
#             x = np.random.uniform(self.xlim[0], self.xlim[1])
#             y = np.random.uniform(self.ylim[0], self.ylim[1])
#             point = Point(x, y)
#             if self.plg_border_shapely.contains(point):
#                 return [x, y]

#     def inRedZone(self, location):
#         x, y = location
#         point = Point(x, y)
#         collision = False
#         for i in range(len(self.plg_obstacle_shapely)):
#             if self.plg_obstacle_shapely[i].contains(point):
#                 collision = True
#         return collision

#     @staticmethod
#     def get_distance_between_nodes(n1, n2):
#         dist_x = n1.x - n2.x
#         dist_y = n1.y - n2.y
#         dist = np.sqrt(dist_x**2 + dist_y**2)
#         return dist

#     # def set_obstacles(self):
#     #     for i in range(self.obstacles.shape[0]):
#     #         self.polygon_obstacles.append(Polygon(list(map(tuple, self.obstacles[i]))))

#     def iscollided(self, n1, n2):
#         line = LineString([(n1.x, n1.y),
#                            (n2.x, n2.y)])
#         collision = False
#         for i in range(len(self.plg_obstacle_shapely)):
#             if self.plg_obstacle_shapely[i].intersects(line):
#                 collision = True
#         return collision

#     def get_shortest_path_using_dijkstra(self):
#         self.unvisited_nodes = []
#         for node in self.nodes:
#             node.cost = np.inf
#             node.parent = None
#             self.unvisited_nodes.append(node)

#         current_node = self.unvisited_nodes[0]
#         current_node.cost = 0
#         pointer_node = current_node

#         while self.unvisited_nodes:
#             ind_min_cost = PRM.get_ind_min_cost(self.unvisited_nodes)
#             current_node = self.unvisited_nodes[ind_min_cost]

#             for neighbour_node in current_node.neighbours:
#                 if neighbour_node in self.unvisited_nodes:
#                     cost = current_node.cost + PRM.get_distance_between_nodes(current_node, neighbour_node)
#                     if cost < neighbour_node.cost:
#                         neighbour_node.cost = cost
#                         neighbour_node.parent = current_node
#             pointer_node = current_node
#             self.unvisited_nodes.pop(ind_min_cost)

#         self.path.append([pointer_node.x, pointer_node.y])

#         while pointer_node.parent is not None:
#             node = pointer_node.parent
#             self.path.append([node.x, node.y])
#             pointer_node = node

#         self.path.append([self.starting_node.x, self.starting_node.y])

#     @staticmethod
#     def get_ind_min_cost(nodes):
#         cost = []
#         for node in nodes:
#             cost.append(node.cost)
#         return cost.index(min(cost))

#     def plot_prm(self):
#         plt.clf()
#         for i in range(self.obstacles.shape[0]):
#             obstacle = np.append(self.obstacles[i], self.obstacles[i][0, :].reshape(1, -1), axis=0)
#             plt.plot(obstacle[:, 0], obstacle[:, 1], 'r-.')

#         for node in self.nodes:
#             if node.neighbours is not None:
#                 for i in range(len(node.neighbours)):
#                     plt.plot([node.location.x, node.neighbours[i].location.x],
#                              [node.location.y, node.neighbours[i].location.y], "-g")
#         path = np.array(self.path)
#         plt.plot(path[:, 0], path[:, 1], "-r")
#         plt.grid()
#         plt.title("prm")
#         plt.show()


# if __name__ == "__main__":
#     starting_loc = [0, 0]
#     ending_loc = [1, 1]

#     MAXNUM = 100
#     XLIM = [0, 1]
#     YLIM = [0, 1]
#     GOAL_SAMPLE_RATE = .01
#     STEP = .1
#     RADIUS_NEIGHBOUR = .15
#     DISTANCE_TOLERANCE = .11
#     # OBSTACLES = [[[.1, .1], [.2, .1], [.2, .2], [.1, .2]],
#     #              [[.4, .4], [.6, .5], [.5, .6], [.3, .4]],
#     #              [[.8, .8], [.95, .8], [.95, .95], [.8, .95]]]
#     OBSTACLES = [[[.1, .0], [.2, .0], [.2, .5], [.1, .5]],
#                  [[.0, .6], [.6, .6], [.6, 1.], [.0, 1.]],
#                  [[.8, .0], [1., .0], [1., .9], [.8, .9]],
#                  [[.3, .1], [.4, .1], [.4, .6], [.3, .6]],
#                  [[.5, .0], [.6, .0], [.6, .4], [.5, .4]]]

#     FIGPATH = os.getcwd() + "/../../fig/prm/"

#     prm = PRM(starting_loc, ending_loc, np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]), np.array(OBSTACLES))
#     prm.set_obstacles()
#     prm.get_all_random_nodes()
#     prm.get_road_maps()
#     prm.get_shortest_path_using_dijkstra()
#     # prm.get_shortest_path_using_astar()
#     prm.plot_prm()
#     pass

import numpy as np
import streamlit as st
from shapely.geometry import Point, Polygon, LineString
import plotly.graph_objects as go

class Node:
    def __init__(self, loc):
        self.x, self.y = loc
        self.cost = np.inf  # Initialize cost to infinity
        self.parent = None
        self.neighbours = []

class PRM:
    def __init__(self, border, obstacles):
        self.nodes = []
        self.polygon_border = border
        self.polygon_obstacles = obstacles
        self.border = Polygon(border)
        self.obstacles = [Polygon(obstacle) for obstacle in obstacles]
        self.animated = False
    
    def get_path(self, loc_start: np.ndarray, loc_end: np.ndarray, num_nodes: int=1000, num_neighbours: int=10, animated: bool=False) -> (list, go.Figure):
        self.nodes = []
        self.path = []
        self.loc_start = loc_start
        self.loc_end = loc_end
        self.num_nodes = num_nodes
        self.num_neighbours = num_neighbours
        self.animated = animated
        fig = self.get_road_map()
        return self.path, fig

    def get_road_map(self) -> go.Figure:
        self.starting_node = Node(self.loc_start)
        self.ending_node = Node(self.loc_end)
        self.starting_node.cost = 0  # Set start node cost to zero
        self.nodes = [self.starting_node] + [Node(self.get_new_location()) for _ in range(self.num_nodes - 2)] + [self.ending_node]

        for i in range(len(self.nodes)):
            node_now = self.nodes[i]
            dists = [(self.get_distance_between_nodes(node_now, node), node) for node in self.nodes if node != node_now]
            dists.sort(key=lambda x: x[0])
            for dist, node_neighbour in dists[:self.num_neighbours]:
                if not self.is_collided(node_now, node_neighbour):
                    node_now.neighbours.append(node_neighbour)
        
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
            loc_nodes = np.array([[node.x, node.y] for node in self.nodes])
            fig.add_trace(go.Scatter(x=loc_nodes[:, 0], y=loc_nodes[:, 1], mode='markers', marker=dict(size=7, color='white')))
            fig.add_trace(go.Scatter(x=[self.loc_start[0]], y=[self.loc_start[1]], mode='markers', marker=dict(size=20, color='green')))
            fig.add_trace(go.Scatter(x=[self.loc_end[0]], y=[self.loc_end[1]], mode='markers', marker=dict(size=20, color='blue')))

            progress_bar = st.progress(0)
            self.chart_placeholder = st.empty()
            for node in self.nodes:
                for neighbour in node.neighbours:
                    fig.add_trace(go.Scatter(x=[node.x, neighbour.x], y=[node.y, neighbour.y], mode='lines', line=dict(color='white', width=.5), opacity=.5))
                    self.chart_placeholder.plotly_chart(fig, use_container_width=True)
                    progress_bar.progress((i + 1) / len(self.nodes))
            progress_bar.empty()
        
        self.get_shortest_path_using_dijkstra()

        if self.animated:
            path = np.array(self.path)
            fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode='lines', line=dict(color='yellow', width=5)))
            self.chart_placeholder.plotly_chart(fig, use_container_width=True)
        else:
            fig = None

        return fig

    def get_shortest_path_using_dijkstra(self):
        unvisited = self.nodes[:]
        while unvisited:
            current_node = min(unvisited, key=lambda node: node.cost)
            unvisited.remove(current_node)
            for neighbour in current_node.neighbours:
                new_cost = current_node.cost + self.get_distance_between_nodes(current_node, neighbour)
                if new_cost < neighbour.cost:
                    neighbour.cost = new_cost
                    neighbour.parent = current_node

        # Trace back the path from end to start
        self.path = []
        trace_node = self.ending_node
        while trace_node:
            self.path.append([trace_node.x, trace_node.y])
            trace_node = trace_node.parent
        self.path.reverse()

    def get_new_location(self):
        while True:
            x = np.random.uniform(self.border.bounds[0], self.border.bounds[2])
            y = np.random.uniform(self.border.bounds[1], self.border.bounds[3])
            point = Point(x, y)
            if self.border.contains(point) and all(not obs.contains(point) for obs in self.obstacles):
                return [x, y]

    def get_distance_between_nodes(self, n1, n2):
        return np.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)

    def is_collided(self, n1, n2):
        line = LineString([(n1.x, n1.y), (n2.x, n2.y)])
        return any(obs.intersects(line) for obs in self.obstacles)




