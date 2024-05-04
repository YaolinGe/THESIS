"""
A* algorithm for path planning.
"""

from shapely.geometry import Polygon, Point, LineString
import streamlit as st
import plotly.graph_objects as go
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import math
# import os
# from unittest import TestCase
# from matplotlib.gridspec import GridSpec


class Node:

    def __init__(self, loc: np.ndarray, parent=None):
        self.x, self.y = loc
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0


class AStar:

    def __init__(self, polygon_border=None, polygon_obstacles=None):
        self.polygon_border = polygon_border
        self.polygon_obstacles = polygon_obstacles
        self.polygon_border_shapely = Polygon(self.polygon_border)
        self.polygon_obstacles_shapely = [Polygon(plg) for plg in self.polygon_obstacles]

    def search_path(self, loc_start, loc_end, max_iter: int=2000, stepsize: float=.05,
                    distance_tolerance_target: float=.075, distance_tolerance: float=.02, animated: bool=False) -> list:
        self.loc_start = loc_start
        self.loc_target = loc_end
        self.cnt = 0
        self.open_list = []
        self.closed_list = []
        self.path = []
        self.arrival = False
        self.animated = animated

        self.max_iter = max_iter
        self.stepsize = stepsize
        self.distance_tolerance_target = distance_tolerance_target
        self.distance_tolerance = distance_tolerance

        angles = np.arange(0, 360, 60)

        # s1: initialise nodes
        self.start_node = Node(loc_start, None)
        self.start_node.g = self.start_node.h = self.start_node.f = 0
        self.end_node = Node(loc_end, None)
        self.end_node.g = self.end_node.h = self.end_node.f = 0
        self.open_list = []
        self.closed_list = []

        # s2: append open list
        self.open_list.append(self.start_node)


        ### Animation
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
            fig.add_trace(go.Scatter(x=[self.loc_start[0]], y=[self.loc_start[1]], mode='markers', marker=dict(size=20, color='green')))
            fig.add_trace(go.Scatter(x=[self.loc_target[0]], y=[self.loc_target[1]], mode='markers', marker=dict(size=20, color='blue')))
            progress_bar = st.progress(0)
            self.chart_placeholder = st.empty()


        # s3: loop open list
        while len(self.open_list) > 0:
            # s31: find smallest cost node and append this to closed list and remove it from open list.
            node_now = self.get_min_cost_node()
            self.closed_list.append(node_now)

            if self.is_arrived(node_now):
                pointer = node_now
                while pointer is not None:
                    self.path.append([pointer.x, pointer.y])
                    pointer = pointer.parent

            # s32: produce children and then start allocating locations.
            children = []
            for angle in angles:
                ang = math.radians(angle)
                xn = node_now.x + np.cos(ang) * self.stepsize
                yn = node_now.y + np.sin(ang) * self.stepsize
                loc_n = np.array([xn, yn])
                if self.is_within_obstacle(loc_n) or not self.is_within_border(loc_n) or not self.is_path_legal(np.array([node_now.x, node_now.y]), loc_n):
                    continue
                node_new = Node(loc_n, node_now)
                children.append(node_new)

            # s33: loop through all children to filter illegal points.
            for child in children:
                if self.is_node_in_list(child, self.closed_list):
                    continue

                child.g = node_now.g + self.stepsize
                child.h = self.get_distance_between_nodes(child, self.end_node)
                child.f = child.g + child.h

                if self.is_node_in_list(child, self.open_list):
                    continue

                self.open_list.append(child)

            if self.animated:
                fig.add_trace(go.Scatter(x=[node.x for node in self.open_list], y=[node.y for node in self.open_list], mode='markers', marker=dict(size=5, color='cyan')))
                fig.add_trace(go.Scatter(x=[node.x for node in self.closed_list], y=[node.y for node in self.closed_list], mode='markers', marker=dict(size=2.5, color='gray'), opacity=.5))
                fig.add_trace(go.Scatter(x=[node_now.x], y=[node_now.y], mode='markers', marker=dict(size=5, color='white')))
                fig.add_trace(go.Scatter(x=[child.x for child in children], y=[child.y for child in children], mode='markers', marker=dict(size=2.5, color='red'), opacity=.5))
                self.chart_placeholder.plotly_chart(fig, use_container_width=True)
                progress_bar.progress(self.cnt / self.max_iter)

            # print("cnt: ", self.cnt)
            self.cnt += 1
            if self.cnt > self.max_iter:
                print("Cannot converge")
                break

            if self.arrival:
                print("Arrived")
                self.path.insert(0, loc_end)
                path = np.array(self.path)
                if self.animated:
                    fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode='lines', line=dict(width=5, color='yellow')))
                    self.chart_placeholder.plotly_chart(fig, use_container_width=True)
                    progress_bar.empty()
                return self.path

    def get_min_cost_node(self):
        min_node = self.open_list[0]
        for node in self.open_list:
            if node.f < min_node.f:
                min_node = node
        self.open_list.remove(min_node)
        return min_node

    def is_arrived(self, node):
        dist = self.get_distance_between_nodes(node, self.end_node)
        if dist <= self.distance_tolerance_target:
            self.arrival = True
            return True
        else:
            return False

    def is_node_in_list(self, node, l):
        for e in l:
            dist = self.get_distance_between_nodes(node, e)
            if dist <= self.distance_tolerance:
                return True
        return False

    def is_within_border(self, loc):
        point = Point(loc[0], loc[1])
        return self.polygon_border_shapely.contains(point)

    def is_within_obstacle(self, loc):
        point = Point(loc[0], loc[1])
        for plg in self.polygon_obstacles_shapely:
            if plg.contains(point):
                return True
        return False

    def is_path_legal(self, loc1: np.ndarray, loc2: np.ndarray) -> bool:
        x1, y1 = loc1
        x2, y2 = loc2
        path = LineString([(x1, y1), (x2, y2)])
        for plg in self.polygon_obstacles_shapely:
            if plg.intersects(path):
                return False
        return True

    @staticmethod
    def get_distance_between_nodes(n1, n2):
        return np.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)


# class TestAstar(TestCase):
#     def setUp(self) -> None:
#         self.plg_border = np.array([[0, 0],
#                                [0, 1],
#                                [1, 1],
#                                [1, 0],
#                                [0, 0]])

#         # self.plg_obstacle = np.array([[.5, .5],
#         #                               [.51, .5],
#         #                               [.51, .51],
#         #                               [.5, .51],
#         #                               [.5, .5]])
#         # self.plg_obstacle = np.array([[.25, .25],
#         #                          [.65, .25],
#         #                          [.65, .65],
#         #                          [.25, .65],
#         #                          [.25, .25]])
#         self.plg_obstacle = np.array([[.21, .21],
#                                       [.41, .21],
#                                       [.41, .61],
#                                       [.61, .61],
#                                       [.61, .21],
#                                       [.81, .21],
#                                       [.81, .81],
#                                       [.21, .81],
#                                       [.21, .21]])
#         self.wp = WaypointGraph()
#         self.wp.set_polygon_border(self.plg_border)
#         self.wp.set_polygon_obstacles([self.plg_obstacle])
#         self.wp.set_depth_layers([0])
#         self.wp.set_neighbour_distance(.05)
#         self.wp.construct_waypoints()
#         self.wp.construct_hash_neighbours()
#         self.waypoint = self.wp.get_waypoints()

#         self.astar = AStar(self.plg_border, self.plg_obstacle)

#     def test_astar(self):
#         loc_start = [.01, .01]
#         loc_end = [.99, .99]
#         self.astar.search_path(loc_start, loc_end)




