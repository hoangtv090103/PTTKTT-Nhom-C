import numpy as np
from shapely import affinity
from shapely.geometry import Point
import solution

RESOLUTION = 16

class Ellipse:
    def __init__(self, center, half_width, half_height, polygon=None):
        if type(center) != Point:
            center = Point(center[0], center[1])

        self.center = center
        self.half_width = half_width
        self.half_height = half_height

        # Approximate ellipse as polygon
        circle_approximation = center.buffer(1, resolution=RESOLUTION)
        if not polygon:
            polygon = affinity.scale(circle_approximation, self.half_width, self.half_height)
        self.polygon = polygon

    def __reduce__(self):
        return Ellipse, (self.center, self.half_width, self.half_height, self.polygon)

    @property
    def bounds(self):
        """Return the bounding box (min_x, min_y, max_x, max_y)"""
        x, y = self.center.x, self.center.y
        return (
            x - self.half_width,  # min_x
            y - self.half_height, # min_y 
            x + self.half_width,  # max_x
            y + self.half_height  # max_y
        )

    def intersects(self, other):
        return solution.do_shapes_intersect(self.polygon, other)

    def within(self, other):
        return solution.does_shape_contain_other(other, self.polygon)

    def contains(self, other):
        return solution.does_shape_contain_other(self.polygon, other)

    @property 
    def area(self):
        return np.pi * self.half_width * self.half_height