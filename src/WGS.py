"""
WGS 84 coordinate system.
North-East-Down reference is employed.

It converts (lat, lon) in degrees to (x, y) in meters given a specific origin.
The selected origin is at Nidarosdomen in Trondheim.

Example:
    >>> wgs = WGS()
    >>> x, y = wgs.latlon2xy(64.55, 10.55)
    >>> print(x, y)
    >>> 11131.944444443812 4783.655665331498
    >>> x, y = 1000, 2000
    >>> lat, lon = wgs.xy2latlon(x, y)
    >>> print(lat, lon)
    >>> 64.45898315658141 10.49166998986048
"""

import numpy as np
from math import degrees, radians
from numpy import vectorize


class WGS:
    __CIRCUMFERENCE = 40075000  # [m], circumference
    __LATITUDE_ORIGIN = 63.4269097
    __LONGITUDE_ORIGIN = 10.3969375

    @staticmethod
    @vectorize
    def latlon2xy(lat, lon):
        x = radians((lat - WGS.__LATITUDE_ORIGIN)) / 2 / np.pi * WGS.__CIRCUMFERENCE
        y = radians((lon - WGS.__LONGITUDE_ORIGIN)) / 2 / np.pi * WGS.__CIRCUMFERENCE * np.cos(radians(lat))
        return x, y

    @staticmethod
    @vectorize
    def xy2latlon(x, y):
        lat = WGS.__LATITUDE_ORIGIN + degrees(x * np.pi * 2.0 / WGS.__CIRCUMFERENCE)
        lon = WGS.__LONGITUDE_ORIGIN + degrees(y * np.pi * 2.0 / (WGS.__CIRCUMFERENCE * np.cos(radians(lat))))
        return lat, lon

    @staticmethod
    def get_origin() -> tuple:
        """ Return origin lat, lon in degrees. """
        return WGS.__LATITUDE_ORIGIN, WGS.__LONGITUDE_ORIGIN

    @staticmethod
    def get_circumference() -> float:
        return WGS.__CIRCUMFERENCE


if __name__ == "__main__":
    wgs = WGS()
    x, y = wgs.latlon2xy(64.55, 10.55)
    print(x, y)
    x, y = 1000, 2000
    lat, lon = wgs.xy2latlon(x, y)
    print(lat, lon)
