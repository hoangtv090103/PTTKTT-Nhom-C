# shape_utils.py

from shapely.geometry import LinearRing, MultiPolygon, Point, Polygon, MultiPoint, MultiLineString


def get_bounds(shape):
    """Return a tuple with the (min_x, min_y, max_x, max_y) describing the bounding box of the shape"""
    if hasattr(shape, 'center') and hasattr(shape, 'radius'):
        return shape.center.x - shape.radius, shape.center.y - shape.radius, shape.center.x + shape.radius, shape.center.y + shape.radius

    if hasattr(shape, 'polygon'):
        shape = shape.polygon

    return shape.bounds


def get_bounding_rectangle_center(shape):
    """Return the center of the bounding rectangle for the passed shape"""
    if hasattr(shape, 'center'):
        return shape.center.x, shape.center.y

    if hasattr(shape, 'polygon'):
        shape = shape.polygon

    return (shape.bounds[0] + shape.bounds[2]) / 2, (shape.bounds[1] + shape.bounds[3]) / 2


def get_centroid(shape):
    """Return the centroid of a shape"""
    if hasattr(shape, 'center'):
        return shape.center

    if hasattr(shape, 'polygon'):
        shape = shape.polygon

    return shape.centroid
