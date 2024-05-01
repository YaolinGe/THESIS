"""
Given a x and y range, generate a path using a fixed step size and a random number of steps. 

Parameters
----------
x_range : tuple of float
    The range of x values.
y_range : tuple of float
    The range of y values.
step_size : float
    The size of the step to take.
number_of_steps : int 
    The number of steps to take.

Returns
-------
numpy.ndarray
    The generated path.
"""

import numpy as np

def generate_a_path(x_range, y_range, step_size, number_of_steps):
    x = np.random.uniform(x_range[0], x_range[1])
    y = np.random.uniform(y_range[0], y_range[1])
    path = np.array([[x, y]])
    for _ in range(number_of_steps):
        angle = np.random.uniform(0, 2 * np.pi)
        x += step_size * np.cos(angle)
        y += step_size * np.sin(angle)
        if x < x_range[0] or x > x_range[1] or y < y_range[0] or y > y_range[1]:
            break
        path = np.vstack((path, [x, y]))
    return path