"""
This function generates a random non-convex polygon given the maximum and minimum x and y coordinates
and the number of vertices.

Args:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    n_vertices: int

Returns:
    polygon: np.array([x, y])
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def generate_polygon_border(x_min: float, x_max: float, y_min: float, y_max: float, n_vertices: int) -> np.array:
    x = np.random.uniform(x_min, x_max, n_vertices)
    y = np.random.uniform(y_min, y_max, n_vertices)
    points = np.array([x, y]).T
    hull = ConvexHull(points)
    polygon = points[hull.vertices]
    return polygon

def generate_polygon_obstacle_inside_polygon_border(polygon_border: np.ndarray):
    """
    This function generates a random non-convex polygon inside a given polygon border.

    Args:
        polygon_border: np.array([x, y])

    Returns:
        polygon_obstacle: np.array([x, y])
    """
    x_min, x_max = np.min(polygon_border[:, 0]), np.max(polygon_border[:, 0])
    y_min, y_max = np.min(polygon_border[:, 1]), np.max(polygon_border[:, 1])
    n_vertices = np.random.randint(3, 10)
    polygon_obstacle = generate_polygon_border(x_min, x_max, y_min, y_max, n_vertices)
    return polygon_obstacle

# Example usage
# polygon = generate_polygon_border(0, 1, 0, 1, 5)
# plt.figure()
# plt.plot(polygon[:, 0], polygon[:, 1], 'r-')
# plt.show()

if __name__ == "__main__":
    polygon = generate_polygon_border(0, 1, 0, 1, 5)
    polygon_obstacle = generate_polygon_obstacle_inside_polygon_border(polygon)
    plt.figure()
    plt.plot(polygon[:, 0], polygon[:, 1], 'r-')
    plt.plot(polygon_obstacle[:, 0], polygon_obstacle[:, 1], 'b-')
    plt.show()

