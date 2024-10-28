import copy
import numpy as np
from shapely import Point
from shapely import affinity

from utils import copy_shape, get_bounding_rectangle_center


class PlacedShape(object):

    """Class representing a geometric shape placed in a container with a reference position and rotation"""

    __slots__ = ("shape", "position", "rotation")

    def __init__(self, shape, position=(0., 0.), rotation=0., move_and_rotate=True):
        """Constructor"""

        # the original shape should remain unchanged, while this version is moved and rotated in the 2D space
        self.shape = copy_shape(shape)

        # the original points of the shape only represented distances among them, now the center of the bounding rectangle of the shape should be found in the reference position
        self.position = position
        if move_and_rotate:
            self.update_position(position)
            bounding_rectangle_center = get_bounding_rectangle_center(self.shape)
            self.move((position[0] - bounding_rectangle_center[0], position[1] - bounding_rectangle_center[1]), False)

        # rotate accordingly to the specified angle
        self.rotation = rotation
        if move_and_rotate:
            self.rotate(rotation, False)

    def __deepcopy__(self, memo=None):
        """Return a deep copy"""

        # the constructor already deep-copies the shape
        return PlacedShape(self.shape, copy.deepcopy(self.position), self.rotation, False)

    def update_position(self, new_position):
        from circle import Circle
        from ellipse import Ellipse

        """Update the position"""

        self.position = new_position

        # update the center for the circle or ellipse
        if type(self.shape) == Circle or type(self.shape) == Ellipse:
            self.shape.center = Point(new_position[0], new_position[1])

    def move(self, displacement, update_reference_position=True):
        from circle import Circle
        from ellipse import Ellipse

        """Move the shape as much as indicated by the displacement"""

        # for the ellipse, apply the action to the approximate polygon
        if type(self.shape) == Ellipse:
            shape_to_move = self.shape.polygon

        # for the circle, convert to polygon approximation
        elif type(self.shape) == Circle:
            shape_to_move = self.shape.to_polygon()

        # otherwise move the shape itself
        else:
            shape_to_move = self.shape

        # only move when it makes sense
        if displacement != (0., 0.) or (type(self.shape) == Ellipse and get_bounding_rectangle_center(shape_to_move) != self.shape.center):

            shape_to_move = affinity.translate(shape_to_move, displacement[0], displacement[1])

            if type(self.shape) == Ellipse:
                self.shape.polygon = shape_to_move

            elif type(self.shape) == Circle:
                self.shape.polygon = shape_to_move

            else:
                self.shape = shape_to_move

            if update_reference_position:
                self.update_position((self.position[0] + displacement[0], self.position[1] + displacement[1]))

        # for the circle, update the support approximate polygon
        if type(self.shape) == Circle:
            center_displacement = self.shape.center.x - self.shape.polygon.centroid.x, self.shape.center.y - self.shape.polygon.centroid.y
            if center_displacement != (0, 0):
                self.shape.polygon = affinity.translate(self.shape.polygon, center_displacement[0], center_displacement[1])

    def move_to(self, new_position):
        """Move the shape to a new position, updating its points"""

        self.move(displacement=(new_position[0] - self.position[0], new_position[1] - self.position[1]))

    def rotate(self, angle, update_reference_rotation=True, origin=None):
        from circle import Circle
        from ellipse import Ellipse

        """Rotate the shape around its reference position according to the passed rotation angle, expressed in degrees"""

        # only rotate when it makes sense
        if not np.isnan(angle) and angle != 0 and (type(self.shape) != Circle or origin is not None):

            # for the ellipse, apply the action to the approximate polygon
            if type(self.shape) == Ellipse:
                shape_to_rotate = self.shape.polygon

            # for the circle, convert to polygon approximation
            elif type(self.shape) == Circle:
                shape_to_rotate = self.shape.to_polygon()

            # otherwise rotate the shape itself
            else:
                shape_to_rotate = self.shape

            if not origin:
                origin = self.position
            shape_to_rotate = affinity.rotate(shape_to_rotate, angle, origin)

            if type(self.shape) == Ellipse:
                self.shape.polygon = shape_to_rotate

            elif type(self.shape) == Circle:
                self.shape.polygon = shape_to_rotate

            else:
                self.shape = shape_to_rotate

            if update_reference_rotation:
                self.rotation += angle

    def rotate_to(self, new_rotation):
        """Rotate the shape around its reference position so that it ends up having the passed new rotation"""

        self.rotate(new_rotation - self.rotation)