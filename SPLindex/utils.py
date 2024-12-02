import numpy as np

from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon




def get_mbb(sorted_cluster):
    # polygons is a list of the 100 polygons in the cluster
    polygons = [Polygon(coords) for coords in sorted_cluster]
    # Convert the list of polygons to a MultiPolygon object
    cluster = MultiPolygon(polygons)
    # Calculate the minimum bounding box of the cluster
    minx, miny, maxx, maxy = cluster.bounds
    mbb = [(minx, miny), (maxx, maxy)]
    # Create a rectangle object from the bounding box coordinates
    #bounding_box = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])
    return mbb


"""def calculate_bounding_box(rectangles):
    x_min = float('inf')
    y_min = float('inf')
    x_max = float('-inf')
    y_max = float('-inf')

    for polygon, rectangle in rectangles:
        x_min = min(x_min, rectangle[0])
        y_min = min(y_min, rectangle[1])
        x_max = max(x_max, rectangle[2])
        y_max = max(y_max, rectangle[3])
    mbb = [(x_min, y_min), (x_max, y_max)]
    return mbb"""

def calculate_bounding_box(rectangles):
    x_min = float('inf')
    y_min = float('inf')
    x_max = float('-inf')
    y_max = float('-inf')

    for rectangle in rectangles:
        x_min = min(x_min, rectangle[0])
        y_min = min(y_min, rectangle[1])
        x_max = max(x_max, rectangle[2])
        y_max = max(y_max, rectangle[3])
    mbb = [(x_min, y_min), (x_max, y_max)]
    return mbb


