"""
Generate a path from start to end using Probabilistic Road Map (PRM) algorithm.
"""
import numpy as np
from shapely.geometry import Point, Polygon, LineString
import heapq

def prm_path_planning(loc_start, loc_end, polygon_border, polygon_obstacles, num_nodes=1000, num_neighbours=10):
    """
    Plan a path using the Probabilistic Roadmap (PRM) algorithm.

    Args:
        loc_start (list): The starting location [x, y].
        loc_end (list): The ending location [x, y].
        polygon_border (numpy.ndarray): The vertices of the border polygon.
        polygon_obstacles (list): A list of vertices for each obstacle polygon.
        num_nodes (int, optional): The number of random nodes to generate. Defaults to 1000.
        num_neighbours (int, optional): The number of nearest neighbors to consider for each node. Defaults to 10.

    Returns:
        list: The path from the starting location to the ending location as a list of waypoints.

    Example:
        path = prm_path_planning([0, 0], [1, 1], np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]), [])
    """
    border_polygon = Polygon(polygon_border)
    obstacle_polygons = [Polygon(obs) for obs in polygon_obstacles]
    xlim, ylim = [np.min(polygon_border[:, 0]), np.max(polygon_border[:, 0])], [np.min(polygon_border[:, 1]), np.max(polygon_border[:, 1])]

    def random_location():
        """
        Generate a random location within the boundary polygon that is not inside any obstacle polygon.

        Returns:
            list: The randomly generated location [x, y].
        """
        while True:
            loc = [np.random.uniform(*xlim), np.random.uniform(*ylim)]
            if border_polygon.contains(Point(*loc)) and not any(Polygon(o).contains(Point(*loc)) for o in polygon_obstacles):
                return loc

    def distance(n1, n2):
        """
        Calculate the Euclidean distance between two nodes.

        Args:
            n1 (list): The coordinates of the first node [x, y].
            n2 (list): The coordinates of the second node [x, y].

        Returns:
            float: The Euclidean distance between the two nodes.
        """
        return np.linalg.norm(np.array(n1) - np.array(n2))

    def valid_edge(n1, n2):
        """
        Check if an edge between two nodes is valid, i.e., it does not intersect with any obstacle polygon.

        Args:
            n1 (list): The coordinates of the first node [x, y].
            n2 (list): The coordinates of the second node [x, y].

        Returns:
            bool: True if the edge is valid, False otherwise.
        """
        return not any(LineString([n1, n2]).intersects(obs) for obs in obstacle_polygons)

    nodes = [loc_start] + [random_location() for _ in range(num_nodes - 2)] + [loc_end]
    roadmap = {tuple(n): [] for n in nodes}

    for n1 in nodes:
        neighbours = sorted(nodes, key=lambda n: distance(n1, n))[:num_neighbours]
        for n2 in neighbours:
            if n1 != n2 and valid_edge(n1, n2):
                roadmap[tuple(n1)].append(n2)

    def dijkstra_path(start, goal):
        """
        Find the shortest path from the starting location to the ending location using Dijkstra's algorithm.

        Args:
            start (list): The starting location [x, y].
            goal (list): The ending location [x, y].

        Returns:
            list: The shortest path from the starting location to the ending location as a list of waypoints.
        """
        pq, costs, parents = [(0, tuple(start))], {tuple(start): 0}, {}
        while pq:
            cost, node = heapq.heappop(pq)
            if node == tuple(goal):
                path, current = [], node
                while current:
                    path.append(list(current))
                    current = parents.get(current)
                return path[::-1]
            for neighbor in roadmap[node]:
                ncost = cost + distance(node, neighbor)
                if ncost < costs.get(tuple(neighbor), float('inf')):
                    costs[tuple(neighbor)] = ncost
                    parents[tuple(neighbor)] = node
                    heapq.heappush(pq, (ncost, tuple(neighbor)))
        return []

    return dijkstra_path(loc_start, loc_end)

# Example usage:
if __name__ == "__main__":
    loc_start = [0, 0]
    loc_end = [1, 1]
    polygon_border = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    polygon_obstacles = [
        [[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.4], [0.2, 0.2]],
        [[0.6, 0.6], [0.8, 0.6], [0.8, 0.8], [0.6, 0.8], [0.6, 0.6]],
        [[0.2, 0.6], [0.4, 0.6], [0.4, 0.8], [0.2, 0.8], [0.2, 0.6]],
        [[0.6, 0.2], [0.8, 0.2], [0.8, 0.4], [0.6, 0.4], [0.6, 0.2]]
    ]
    path = prm_path_planning(loc_start, loc_end, polygon_border, polygon_obstacles)
    import matplotlib.pyplot as plt
    plt.plot(polygon_border[:, 0], polygon_border[:, 1], 'r-.')
    plt.plot(loc_start[0], loc_start[1], 'bo')
    plt.plot(loc_end[0], loc_end[1], 'go')
    for obs in polygon_obstacles:
        plt.plot(np.array(obs)[:, 0], np.array(obs)[:, 1], 'r-.')
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], 'b-')
    plt.gca().set_aspect('equal')
    plt.show()


