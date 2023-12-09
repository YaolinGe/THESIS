"""
Unit test for Config.py

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-08-22
"""

from unittest import TestCase
from src.Config import Config
import matplotlib.pyplot as plt


class TestConfig(TestCase):
    """
    A TestCase class for testing the Config class.

    Methods
    -------
    setUp():
        Initializes the Config object for each test method.
    test_starting_home_location():
        Tests the starting home location functionality by plotting the location start, location end, and
        polygon borders and obstacles using the Config class methods.
    test_wgs_starting_home_location():
        Tests the WGS starting home location functionality by plotting the location start, location end,
        and polygon borders and obstacles using the Config class methods.
    """

    def setUp(self) -> None:
        """
        Initializes the Config object for each test method.
        """
        self.c = Config()

    def test_starting_home_location(self):
        """
        Tests the starting home location functionality by plotting the location start, location end, and
        polygon borders and obstacles using the Config class methods.
        """
        loc_start = self.c.get_loc_start()
        plg_border = self.c.get_polygon_border()
        plt.plot(plg_border[:, 1], plg_border[:, 0], 'r-.')
        plt.plot(loc_start[1], loc_start[0], 'k.')
        plt.show()
