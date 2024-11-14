import random

from matplotlib import colorbar, pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy import interpolate
from shapely import MultiPolygon, Polygon
from utils import *
from placed_shape import PlacedShape
from shapely.geometry import LineString


class Solution:
    def __init__(self, problem, placed_items=None, weight=0, value=0):
        self.problem = problem
        self.placed_items = placed_items if placed_items else dict()
        self.weight = weight
        self.value = value

    def is_valid_placement(self, item_index):
        """Return whether this solution is valid considering only the item with the specified index and its relation with the rest of items, which is the case only when the placed items do not exceed the capacity of the container, and there are no intersections between items or between an item and the container"""

        # the weight of the item must not cause an exceed of the container's capacity
        if self.weight > self.problem.container.max_weight:
            return False

        shape = self.placed_items[item_index].shape

        # the item must be completely contained in the container
        if not does_shape_contain_other(self.problem.container.shape, shape):
            return False

        # the item's shape is not allowed to intersect with any other placed item's shape
        for other_index, other_placed_shape in self.placed_items.items():
            if item_index != other_index and do_shapes_intersect(shape, other_placed_shape.shape):
                return False

        return True

    def get_area(self):
        """Return the total area of the placed items"""
        return sum([item.shape.area for item in self.placed_items if hasattr(item, 'shape')])

    def get_global_bounds(self):
        """Trả về các giới hạn toàn cầu của hình dạng, bao gồm cả các hình dạng được đặt và không được đặt"""

        global_min_x = global_min_y = np.inf
        global_max_x = global_max_y = -np.inf

        for _, placed_shape in self.placed_items.items():

            min_x, min_y, max_x, max_y = get_bounds(placed_shape.shape)

            if min_x < global_min_x:
                global_min_x = min_x

            if min_y < global_min_y:
                global_min_y = min_y

            if max_x > global_max_x:
                global_max_x = max_x

            if max_y > global_max_y:
                global_max_y = max_y

        return global_min_x, global_min_y, global_max_x, global_max_y

    def get_global_bounding_rectangle_area(self):
        """Trả về diện tích của hình chữ nhật giới hạn toàn cầu của tất cả các hình dạng, bao gồm cả các hình dạng được đặt và không"""

        # Tìm các giới hạn toàn cầu của hình dạng
        min_x, min_y, max_x, max_y = self.get_global_bounds()

        # Trả về diện tích của hình chữ nhật giới hạn bao quanh
        return abs(min_x - max_x) * abs(min_x - max_y)

    def get_random_placed_item_index(self, indices_to_ignore=None):
        """Chọn ngẫu nhiên và trả về một chỉ số của một mục được đặt, loại trừ những mục cần bỏ qua"""

        # lấy các chỉ số của các mục được đặt, loại bỏ những mục cần bỏ qua
        if not indices_to_ignore:
            valid_placed_item_indices = list(self.placed_items.keys())
        else:
            valid_placed_item_indices = [item_index for item_index in self.placed_items.keys(
            ) if item_index not in indices_to_ignore]

        if not valid_placed_item_indices:
            return None

        # Trả về một chỉ số được chọn ngẫu nhiên
        return random.choice(valid_placed_item_indices)

    def _add_item(self, item_index, position, rotation):
        """Place the problem's item with the specified index in the container in the passed position and having the specified rotation, without checking if it leads to an invalid solution"""

        # the item is marked as placed, storing information about the position and rotation of the shape
        self.placed_items[item_index] = PlacedShape(
            self.problem.items[item_index].shape, position, rotation)

        # update the weight and value of the container in the current solution
        self.weight += self.problem.items[item_index].weight
        self.value += self.problem.items[item_index].value

    def add_item(self, item_index, position, rotation=np.nan):
        from circle import Circle

        """Attempt to place the problem's item with the specified index in the container in the passed position and having the specified rotation, and return whether it was possible or otherwise would have lead to an invalid solution"""

        # the index of the item must be valid and the item cannot be already present in the container
        if 0 <= item_index < len(self.problem.items) and item_index not in self.placed_items:

            item = self.problem.items[item_index]

            # the weight of the item must not cause an exceed of the container's capacity
            if self.weight + item.weight <= self.problem.container.max_weight:

                # if the item is a circle, rotation is not relevant
                if not np.isnan(rotation) and type(item.shape) == Circle:
                    rotation = np.nan

                # temporarily insert the item in the container, before intersection checks
                self._add_item(item_index, position, rotation)

                # ensure that the solution is valid with the new placement, i.e. it causes no intersections
                if self.is_valid_placement(item_index):

                    return True

                # undo the placement if it makes the solution unfeasible
                else:

                    self.remove_item(item_index)

        return False

    def remove_item(self, item_index):
        """Attempt to remove the item with the passed index from the container, and return whether it was possible, i.e. whether the item was present in the container before removal"""

        if item_index in self.placed_items:

            # stop considering the weight and value of the item to remove
            self.weight -= self.problem.items[item_index].weight
            self.value -= self.problem.items[item_index].value

            # the item stops being placed
            del self.placed_items[item_index]

            return True

        return False

    def remove_random_item(self):
        """Attempt to remove one of the placed items from the container, selecting it randomly, and return the index of the removed item, or -1 if the container is empty"""

        # if the container is empty, an item index cannot be returned
        if self.weight > 0:

            # choose an index randomly
            removal_index = self.get_random_placed_item_index()

            # perform the removal
            if self.remove_item(removal_index):
                return removal_index

        return -1

    def _move_item(self, item_index, displacement, has_checked_item_in_container=False):
        """Move the item with the passed index as much as indicated by the displacement, without checking if it leads to an invalid solution"""

        if has_checked_item_in_container or item_index in self.placed_items:

            self.placed_items[item_index].move(displacement)

    def move_item(self, item_index, displacement):
        """Attempt to move the item with the passed index as much as indicated by the displacement, and return whether it was possible"""

        if item_index in self.placed_items:

            old_position = self.placed_items[item_index].position

            # temporarily move the item, before intersection checks
            self._move_item(item_index, displacement, True)

            # ensure that the solution is valid with the new movement, i.e. it causes no intersections
            if self.is_valid_placement(item_index):

                return True

            # undo the movement if it makes the solution unfeasible
            else:

                self._move_item_to(item_index, old_position, True)

        return False

    def _move_item_to(self, item_index, new_position, has_checked_item_in_container=False):
        """Move the item with the passed index to the indicated new position, without checking if it leads to an invalid solution"""

        if has_checked_item_in_container or item_index in self.placed_items:

            self.placed_items[item_index].move_to(new_position)

    def move_item_to(self, item_index, new_position):
        """Attempt to move the item with the passed index to the indicated new position, and return whether it was possible"""

        if item_index in self.placed_items:

            old_position = self.placed_items[item_index].position

            # temporarily move the item, before intersection checks
            self._move_item_to(item_index, new_position)

            # ensure that the solution is valid with the new movement, i.e. it causes no intersections
            if self.is_valid_placement(item_index):

                return True

            # undo the movement if it makes the solution unfeasible
            else:

                self._move_item_to(item_index, old_position)

        return False

    def move_item_in_direction(self, item_index, direction, point_num, min_dist_to_check, max_dist_to_check, has_checked_item_in_container=False):
        """Try to move the item with the passed index in the passed (x, y) direction, as far as possible without intersecting, checking as many points as indicated"""

        # at least one point should be checked
        if point_num >= 1:

            if has_checked_item_in_container or item_index in self.placed_items:

                placed_item = self.placed_items[item_index]

                # normalize the direction
                norm = np.linalg.norm(direction)
                direction = (direction[0] / norm, direction[1] / norm)

                # create a line that goes through the reference position of the item and has the passed direction
                line = LineString([placed_item.position, (placed_item.position[0] + direction[0]
                                  * max_dist_to_check, placed_item.position[1] + direction[1] * max_dist_to_check)])

                # find the intersection points of the line with other placed items or the container
                intersection_points = list()
                intersection_points.extend(get_intersection_points_between_shapes(
                    line, self.problem.container.shape))
                for other_index, other_placed_shape in self.placed_items.items():
                    if item_index != other_index:
                        intersection_points.extend(
                            get_intersection_points_between_shapes(line, other_placed_shape.shape))

                # at least an intersection should exist
                if intersection_points:

                    # find the smallest euclidean distance from the item's reference position to the first point of intersection
                    intersection_point, min_dist = min([(p, np.linalg.norm((placed_item.position[0] - p[0], placed_item.position[1] - p[1])))
                                                       for p in intersection_points], key=lambda t: t[1])

                    # only proceed if the two points are not too near
                    if min_dist >= min_dist_to_check:
                        points_to_check = list()

                        # if there is only one point to check, just try that one
                        if point_num == 1:
                            return self.move_item_to(item_index, intersection_point)

                        # the segment between the item's reference position and the nearest intersection is divided in a discrete number of points
                        iter_dist = min_dist / point_num
                        for i in range(point_num - 1):
                            points_to_check.append(
                                (placed_item.position[0] + direction[0] * i * iter_dist, placed_item.position[1] + direction[1] * i * iter_dist))
                        points_to_check.append(intersection_point)

                        # perform binary search to find the furthest point (among those to check) where the item can be placed in a valid way; binary search code is based on bisect.bisect_left from standard Python library, but adapted to perform placement attempts
                        has_moved = False
                        nearest_point_index = 1
                        furthest_point_index = len(points_to_check)
                        while nearest_point_index < furthest_point_index:
                            middle_point_index = (
                                nearest_point_index + furthest_point_index) // 2
                            if self.move_item_to(item_index, points_to_check[middle_point_index]):
                                nearest_point_index = middle_point_index + 1
                                has_moved = True
                            else:
                                furthest_point_index = middle_point_index

                        return has_moved

        return False

    def _rotate_item(self, item_index, angle, has_checked_item_in_container=False, rotate_internal_items=False):
        """Xoay vật phẩm có chỉ số được truyền vào quanh vị trí tham chiếu của nó theo góc xoay được truyền vào, được biểu thị bằng độ, không kiểm tra xem nó có dẫn đến giải pháp không hợp lệ không"""

        if has_checked_item_in_container or item_index in self.placed_items:

            self.placed_items[item_index].rotate(angle)

            # if needed, also rotate any items contained in the item of the passed index, with the origin of the shape containing them
            if rotate_internal_items:

                internal_item_indices = self.get_items_inside_item(item_index)

                for internal_index in internal_item_indices:

                    self.placed_items[internal_index].rotate(
                        angle, False, self.placed_items[item_index].position)

    def rotate_item(self, item_index, angle, rotate_internal_items=False):
        """Cố gắng xoay vật phẩm có chỉ số được truyền vào quanh vị trí tham chiếu của nó theo góc xoay được truyền vào, được biểu thị bằng độ, và trả về xem nó có thể không"""

        if item_index in self.placed_items:

            old_rotation = self.placed_items[item_index].rotation

            # temporarily rotate the item, before intersection checks
            self._rotate_item(item_index, angle, True, rotate_internal_items)

            # ensure that the solution is valid with the new rotation, i.e. it causes no intersections
            if self.is_valid_placement(item_index):

                return True

            # undo the rotation if it makes the solution unfeasible
            else:

                self._rotate_item_to(
                    item_index, old_rotation, True, rotate_internal_items)

        return False

    def _rotate_item_to(self, item_index, new_rotation, has_checked_item_in_container=False, rotate_internal_items=False):
        """Rotate the shape around its reference position so that it ends up having the passed new rotation, without checking if it leads to an invalid solution"""

        if has_checked_item_in_container or item_index in self.placed_items:

            old_rotation = self.placed_items[item_index].rotation

            self.placed_items[item_index].rotate_to(new_rotation)

            # if needed, also rotate any items contained in the item of the passed index, with the origin of the shape containing them
            if rotate_internal_items:

                internal_item_indices = self.get_items_inside_item(item_index)

                for internal_index in internal_item_indices:

                    self.placed_items[internal_index].rotate(
                        new_rotation - old_rotation, False, self.placed_items[item_index].position)

    def rotate_item_to(self, item_index, new_rotation, rotate_internal_items=False):
        """Rotate the shape around its reference position so that it ends up having the passed new rotation, and return whether it was possible"""

        if item_index in self.placed_items:

            old_rotation = self.placed_items[item_index].rotation

            # temporarily rotate the item, before intersection checks
            self._rotate_item_to(item_index, new_rotation,
                                 rotate_internal_items)

            # ensure that the solution is valid with the new rotation, i.e. it causes no intersections
            if self.is_valid_placement(item_index):

                return True

            # undo the rotation if it makes the solution unfeasible
            else:

                self._rotate_item_to(
                    item_index, old_rotation, rotate_internal_items)

        return False

    def rotate_item_in_direction(self, item_index, clockwise, angle_num):
        """Try to rotate the item with the passed index in clockwise or counter-clockwise direction (as specified), checking the maximum number of equally distributed angles as indicated"""
        """Cố gắng xoay vật phẩm có chỉ số được truyền vào theo chiều kim đồng hồ hoặc ngược chiều kim đồng hồ (như được chỉ định), kiểm tra số lượng góc phân bố tối đa như đã chỉ định"""

        has_rotated = False

        if item_index in self.placed_items:

            # calculate the increment in the angle to perform each iteration, to progressively go from an angle greater than 0 to another smaller than 360 (same, and not worth checking since it is the initial state)
            iter_angle = (1 if clockwise else -1) * 360 / (angle_num + 2)
            for _ in range(angle_num):

                # stop as soon as one of the incremental rotations fail; the operation is considered successful if at least one rotation was applied
                if not self.rotate_item(item_index, iter_angle):
                    return has_rotated
                has_rotated = True

        return has_rotated

    def move_and_rotate_item(self, item_index, displacement, angle):
        """Try to move the item with the passed index according to the placed displacement and rotate it as much as indicated by the passed angle"""

        if item_index in self.placed_items:

            old_position = self.placed_items[item_index].position
            old_rotation = self.placed_items[item_index].rotation

            # temporarily move and rotate the item, before intersection checks
            self._move_item(item_index, displacement, True)
            self._rotate_item(item_index, angle, True)

            # ensure that the solution is valid with the new movement and rotation, i.e. it causes no intersections
            if self.is_valid_placement(item_index):

                return True

            # undo the movement and rotation if it makes the solution unfeasible
            else:

                self._move_item_to(item_index, old_position, True)
                self._rotate_item_to(item_index, old_rotation, True)

        return False

    def move_and_rotate_item_to(self, item_index, new_position, new_rotation):
        """Try to move and rotate the item with the passed index so that it has the indicated position and rotation"""

        if item_index in self.placed_items:

            old_position = self.placed_items[item_index].position
            old_rotation = self.placed_items[item_index].rotation

            # temporarily move and rotate the item, before intersection checks
            self._move_item_to(item_index, new_position, True)
            self._rotate_item_to(item_index, new_rotation, True)

            # ensure that the solution is valid with the new movement and rotation, i.e. it causes no intersections
            if self.is_valid_placement(item_index):

                return True

            # undo the movement and rotation if it makes the solution unfeasible
            else:

                self._move_item_to(item_index, old_position, True)
                self._rotate_item_to(item_index, old_rotation, True)

        return False

    def swap_placements(self, item_index0, item_index1, swap_position=True, swap_rotation=True):
        """Try to swap the position and/or the rotation of the two items with the passed indices"""

        # at least position and rotation should be swapped
        if swap_position or swap_rotation:

            # the two items need to be different and placed in the container
            if item_index0 != item_index1 and item_index0 in self.placed_items and item_index1 in self.placed_items:

                # keep track of the original position and rotation of the items
                item0_position = self.placed_items[item_index0].position
                item1_position = self.placed_items[item_index1].position
                item0_rotation = self.placed_items[item_index0].rotation
                item1_rotation = self.placed_items[item_index1].rotation

                # swap position if needed, without checking for validity
                if swap_position:

                    self._move_item_to(item_index0, item1_position, True)
                    self._move_item_to(item_index1, item0_position, True)

                # swap rotation if needed, without checking for validity
                if swap_rotation:

                    self._rotate_item_to(item_index0, item1_rotation, True)
                    self._rotate_item_to(item_index1, item0_rotation, True)

                # ensure that the solution is valid with the swapped movement and/or rotation, i.e. it causes no intersections
                if self.is_valid_placement(item_index0) and self.is_valid_placement(item_index1):

                    return True

                # undo the movement and rotation if it makes the solution unfeasible
                else:

                    # restore position if it was changed
                    if swap_position:

                        self._move_item_to(item_index0, item0_position, True)
                        self._move_item_to(item_index1, item1_position, True)

                    # restore rotation if it was changed
                    if swap_rotation:

                        self._rotate_item_to(item_index0, item0_rotation, True)
                        self._rotate_item_to(item_index1, item1_rotation, True)

        return False

    def get_items_inside_item(self, item_index):
        """Return the indices of the items that are inside the item with the passed index"""

        inside_item_indices = list()

        if item_index in self.placed_items:

            item = self.placed_items[item_index]

            # only multi-polygons can contain other items
            if type(item.shape) == MultiPolygon:

                holes = list()
                for geom in item.shape.geoms:
                    holes.extend(Polygon(hole) for hole in geom.interiors)

                for other_index, placed_shape in self.placed_items.items():

                    if other_index != item_index:

                        for hole in holes:

                            if does_shape_contain_other(hole, self.placed_items[other_index].shape):

                                inside_item_indices.append(other_index)
                                break

        return inside_item_indices

    # def visualize(self, title_override=None, show_title=True, show_container_value_and_weight=True, show_outside_value_and_weight=True, show_outside_items=True, color_items_by_profit_ratio=True, show_item_value_and_weight=True, show_value_and_weight_for_container_items=False, show_reference_positions=False, show_bounding_boxes=False, show_value_weight_ratio_bar=True, force_show_color_bar_min_max=False, show_plot=True, save_path=None):
    #     """Visualize the solution, with placed items in their real position and rotation, and the other ones visible outside the container"""

    #     can_consider_weight = self.problem.container.max_weight != np.inf

    #     # set up the plotting figure
    #     fig_size = (13, 6.75)
    #     dpi = 160
    #     fig = plt.figure(figsize=fig_size, dpi=dpi)
    #     if show_outside_items:
    #         ax1 = fig.add_subplot(1, 2, 1)
    #         ax1.set(aspect="equal")
    #         ax2 = fig.add_subplot(1, 2, 2, sharex=ax1, sharey=ax1)
    #         ax2.set(aspect="equal")
    #         ax2.tick_params(axis="both", which="major", labelsize=11)
    #     else:
    #         ax1 = plt.gca()
    #         ax1.set(aspect="equal")
    #         ax2 = None
    #     ax1.tick_params(axis="both", which="major", labelsize=11)
    #     if show_title:
    #         fig.suptitle(title_override if title_override else "2D Irregular Shape Packing + 0/1 Knapsack Problem")

    #     outside_item_bounds = dict()
    #     total_outside_item_width = 0.

    #     # represent the container
    #     x, y = get_shape_exterior_points(self.problem.container.shape, True)
    #     container_color = (.8, .8, .8)
    #     boundary_color = (0., 0., 0.)
    #     ax1.plot(x, y, color=boundary_color, linewidth=1)
    #     ax1.fill(x, y, color=container_color)
    #     empty_color = (1., 1., 1.)
    #     if type(self.problem.container.shape) == MultiPolygon:
    #         for geom in self.problem.container.shape.geoms:
    #             for hole in geom.interiors:
    #                 x, y = get_shape_exterior_points(hole, True)
    #                 fill_color = empty_color
    #                 boundary_color = (0., 0., 0.)
    #                 ax1.plot(x, y, color=boundary_color, linewidth=1)
    #                 ax1.fill(x, y, color=fill_color)

    #     font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 12}

    #     # show the total value and weight in the container, and the maximum acceptable weight (capacity)
    #     if show_container_value_and_weight:
    #         value_weight_string = "V={}".format(self.value if can_consider_weight else int(self.value))
    #         if can_consider_weight:
    #             value_weight_string += ", W={}, Wmax={}".format(self.weight, self.problem.container.max_weight)
    #         ax1.set_title("Items inside the container\n({})".format(value_weight_string), fontsize=13)

    #     # determine the range of item profitability ratio, for later coloring of items
    #     min_profit_ratio = np.inf
    #     max_profit_ratio = -np.inf
    #     item_profit_ratios = dict()
    #     for item_index, item in enumerate(self.problem.items):
    #         if item.weight == 0:
    #             profit_ratio = np.inf
    #         else:
    #             profit_ratio = item.value / item.weight
    #         item_profit_ratios[item_index] = profit_ratio
    #         min_profit_ratio = min(min_profit_ratio, profit_ratio)
    #         max_profit_ratio = max(max_profit_ratio, profit_ratio)
    #     best_profit_color = (1, 0.35, 0)
    #     worst_profit_color = (1, 0.8, 0.8)
    #     color_interp = interpolate.interp1d([min_profit_ratio, max_profit_ratio], [0, 1])

    #     # if possible, add a color-bar showing the value/weight ratio scale
    #     if show_value_weight_ratio_bar:
    #         fig.subplots_adjust(bottom=0.15)
    #         fig.subplots_adjust(wspace=0.11)
    #         bar_x, bar_y, bar_width, bar_height = 0.5, 0.1, 0.3, 0.02
    #         bar_ax = fig.add_axes([bar_x - bar_width * 0.5, bar_y - bar_height * 0.5, bar_width, bar_height])
    #         color_map = LinearSegmentedColormap.from_list(name="profit-colors", colors=[worst_profit_color, best_profit_color])
    #         norm = colors.Normalize(vmin=min_profit_ratio, vmax=max_profit_ratio)
    #         if force_show_color_bar_min_max:
    #             ticks = np.linspace(min_profit_ratio, max_profit_ratio, 7, endpoint=True)
    #         else:
    #             ticks = None
    #         bar = colorbar.ColorbarBase(bar_ax, cmap=color_map, norm=norm, ticks=ticks, orientation='horizontal', ticklocation="bottom")
    #         bar.set_label(label="value/weight ratio", size=13)
    #         bar.ax.tick_params(labelsize=11)

    #     for index, item in enumerate(self.problem.items):

    #         # represent the placed items
    #         if item_index in self.placed_items:

    #             if color_items_by_profit_ratio:
    #                 fill_color = worst_profit_color + tuple(best_profit_color[i] - worst_profit_color[i]
    #                                                         for i in range(len(best_profit_color))) * color_interp(item_profit_ratios[item_index])
    #             else:
    #                 fill_color = (1, 0.5, 0.5)

    #             self.show_item(item_index, ax1, boundary_color, fill_color, container_color,
    #                            show_item_value_and_weight and show_value_and_weight_for_container_items, font, show_bounding_boxes, show_reference_positions)

    #         # determine the boundary rectangle of the outside-of-container items
    #         elif show_outside_items and ax2:

    #             outside_item_bounds[item_index] = get_bounds(self.problem.items[item_index].shape)
    #             total_outside_item_width += abs(outside_item_bounds[item_index][2] - outside_item_bounds[item_index][0])

    #     # show the outside-of-container items
    #     if show_outside_items and ax2:

    #         out_value_sum = 0
    #         out_weight_sum = 0
    #         row_num = max(1, int(np.log10(len(self.problem.items)) * (3 if len(self.problem.items) < 15 else 4)))
    #         row = 0
    #         width = 0
    #         max_width = 0
    #         row_height = 0
    #         height = 0
    #         for item_index, bounds in outside_item_bounds.items():

    #             out_value_sum += self.problem.items[item_index].value
    #             out_weight_sum += self.problem.items[item_index].weight

    #             if color_items_by_profit_ratio:
    #                 fill_color = worst_profit_color + tuple(best_profit_color[i] - worst_profit_color[i]
    #                                                         for i in range(len(best_profit_color))) * color_interp(item_profit_ratios[item_index])
    #             else:
    #                 fill_color = (1, 0.5, 0.5)

    #             min_x, min_y, max_x, max_y = bounds
    #             shape_width = abs(max_x - min_x)
    #             shape_height = abs(max_y - min_y)

    #             shape_center = get_bounding_rectangle_center(self.problem.items[item_index].shape)
    #             position_offset = (width + shape_width * 0.5 - shape_center[0], row_height + shape_height * 0.5 - shape_center[1])
    #             self.show_item(item_index, ax2, boundary_color, fill_color, empty_color, show_item_value_and_weight,
    #                            font, show_bounding_boxes, show_reference_positions, position_offset)

    #             height = max(height, row_height + shape_height)

    #             width += shape_width
    #             max_width += width
    #             if width >= total_outside_item_width / row_num:
    #                 row += 1
    #                 width = 0
    #                 row_height = height

    #         # show the value and weight outside the container
    #         if show_outside_value_and_weight and ax2:
    #             value_weight_string = "V={}".format(out_value_sum if can_consider_weight else int(out_value_sum))
    #             if can_consider_weight:
    #                 value_weight_string += ", W={}".format(out_weight_sum)
    #             ax2.set_title("Items outside the container\n({})".format(value_weight_string), fontsize=13)

    #     fig = plt.gcf()

    #     if show_plot:
    #         plt.show()

    #     if save_path:
    #         fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    #         plt.close(fig)

    # def show_item(self, item_index, ax, boundary_color, fill_color, container_color, show_item_value_and_weight=False, font=None, show_bounding_box=False, show_reference_position=False, position_offset=(0, 0)):
    #     """Show the shape of the passed item index in the indicated axis with the passed colors"""

    #     if item_index in self.placed_items:
    #         placed_shape = self.placed_items[item_index]
    #         shape = placed_shape.shape
    #     else:
    #         placed_shape = None
    #         shape = self.problem.items[item_index].shape

    #     x, y = get_shape_exterior_points(shape, True)
    #     if position_offset != (0, 0):
    #         x = [x_i + position_offset[0] for x_i in x]
    #         y = [y_i + position_offset[1] for y_i in y]

    #     ax.plot(x, y, color=boundary_color, linewidth=1)
    #     ax.fill(x, y, color=fill_color)

    #     if type(shape) == MultiPolygon:
    #         for geom in shape.geoms:
    #             for hole in geom.interiors:
    #                 x, y = get_shape_exterior_points(hole, True)
    #                 if position_offset != (0, 0):
    #                     x = [x_i + position_offset[0] for x_i in x]
    #                     y = [y_i + position_offset[1] for y_i in y]
    #                 fill_color = container_color
    #                 boundary_color = (0., 0., 0.)
    #                 ax.plot(x, y, color=boundary_color, linewidth=1)
    #                 ax.fill(x, y, color=fill_color)

    #     # show the value and weight in the centroid if required
    #     if show_item_value_and_weight and font:
    #         centroid = get_centroid(shape)
    #         value = self.problem.items[item_index].value
    #         if value != 0 and value == int(value):
    #             value = int(value)
    #         weight = self.problem.items[item_index].weight
    #         if weight != 0 and weight == int(weight):
    #             weight = int(weight)
    #         value_weight_string = "v={}\nw={}".format(value, weight)
    #         item_font = dict(font)
    #         item_font['size'] = 9
    #         ax.text(centroid.x + position_offset[0], centroid.y + position_offset[1], value_weight_string,
    #                 horizontalalignment='center', verticalalignment='center', fontdict=item_font)

    #     # show the bounding box and its center if needed
    #     if show_bounding_box:
    #         bounds = get_bounds(shape)
    #         min_x, min_y, max_x, max_y = bounds
    #         x, y = (min_x, max_x, max_x, min_x, min_x), (min_y, min_y, max_y, max_y, min_y)
    #         if position_offset != (0, 0):
    #             x = [x_i + position_offset[0] for x_i in x]
    #             y = [y_i + position_offset[1] for y_i in y]
    #         boundary_color = (0.5, 0.5, 0.5)
    #         ax.plot(x, y, color=boundary_color, linewidth=1)
    #         bounds_center = get_bounding_rectangle_center(shape)
    #         ax.plot(bounds_center[0] + position_offset[0], bounds_center[1] + position_offset[1], "r.")

    #     # show the reference position if required
    #     if show_reference_position and placed_shape:
    #         ax.plot(placed_shape.position[0], placed_shape.position[1], "b+")

    def visualize(self, title_override=None, show_title=True, show_container_value_and_weight=True, show_outside_value_and_weight=True, show_outside_items=True, color_items_by_profit_ratio=True, show_item_value_and_weight=True, show_value_and_weight_for_container_items=False, show_reference_positions=False, show_bounding_boxes=False, show_value_weight_ratio_bar=True, force_show_color_bar_min_max=False, show_plot=True, save_path=None):
        """Visualize the solution, with placed items in their real position and rotation, and the other ones visible outside the container"""

        can_consider_weight = self.problem.container.max_weight != np.inf

        # set up the plotting figure
        fig_size = (13, 6.75)
        dpi = 160
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        if show_outside_items:
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.set(aspect="equal")
            ax2 = fig.add_subplot(1, 2, 2, sharex=ax1, sharey=ax1)
            ax2.set(aspect="equal")
            ax2.tick_params(axis="both", which="major", labelsize=11)
        else:
            ax1 = plt.gca()
            ax1.set(aspect="equal")
            ax2 = None
        ax1.tick_params(axis="both", which="major", labelsize=11)
        if show_title:
            fig.suptitle(
                title_override if title_override else "2D Irregular Shape Packing + 0/1 Knapsack Problem")

        outside_item_bounds = dict()
        total_outside_item_width = 0.

        # represent the container
        x, y = get_shape_exterior_points(self.problem.container.shape, True)
        container_color = (.8, .8, .8)
        boundary_color = (0., 0., 0.)
        ax1.plot(x, y, color=boundary_color, linewidth=1)
        ax1.fill(x, y, color=container_color)
        empty_color = (1., 1., 1.)
        if type(self.problem.container.shape) == MultiPolygon:
            for geom in self.problem.container.shape.geoms:
                for hole in geom.interiors:
                    x, y = get_shape_exterior_points(hole, True)
                    fill_color = empty_color
                    boundary_color = (0., 0., 0.)
                    ax1.plot(x, y, color=boundary_color, linewidth=1)
                    ax1.fill(x, y, color=fill_color)

        font = {'family': 'serif', 'color':  'black',
                'weight': 'normal', 'size': 12}

        # show the total value and weight in the container, and the maximum acceptable weight (capacity)
        if show_container_value_and_weight:
            value_weight_string = "V={}".format(
                self.value if can_consider_weight else int(self.value))
            if can_consider_weight:
                value_weight_string += ", W={}, Wmax={}".format(
                    self.weight, self.problem.container.max_weight)
            ax1.set_title("Items inside the container\n({})".format(
                value_weight_string), fontsize=13)

        # determine the range of item profitability ratio, for later coloring of items
        min_profit_ratio = np.inf
        max_profit_ratio = -np.inf
        item_profit_ratios = dict()
        for item_index, item in enumerate(self.problem.items):
            if item.weight == 0:
                profit_ratio = np.inf
            else:
                profit_ratio = item.value / item.weight
            item_profit_ratios[item_index] = profit_ratio
            min_profit_ratio = min(min_profit_ratio, profit_ratio)
            max_profit_ratio = max(max_profit_ratio, profit_ratio)
        best_profit_color = (1, 0.35, 0)
        worst_profit_color = (1, 0.8, 0.8)
        color_interp = interpolate.interp1d(
            [min_profit_ratio, max_profit_ratio], [0, 1])

        # if possible, add a color-bar showing the value/weight ratio scale
        if show_value_weight_ratio_bar:
            fig.subplots_adjust(bottom=0.15)
            fig.subplots_adjust(wspace=0.11)
            bar_x, bar_y, bar_width, bar_height = 0.5, 0.1, 0.3, 0.02
            bar_ax = fig.add_axes(
                [bar_x - bar_width * 0.5, bar_y - bar_height * 0.5, bar_width, bar_height])
            color_map = LinearSegmentedColormap.from_list(
                name="profit-colors", colors=[worst_profit_color, best_profit_color])
            norm = colors.Normalize(
                vmin=min_profit_ratio, vmax=max_profit_ratio)
            if force_show_color_bar_min_max:
                ticks = np.linspace(
                    min_profit_ratio, max_profit_ratio, 7, endpoint=True)
            else:
                ticks = None
            bar = colorbar.ColorbarBase(bar_ax, cmap=color_map, norm=norm,
                                        ticks=ticks, orientation='horizontal', ticklocation="bottom")
            bar.set_label(label="value/weight ratio", size=13)
            bar.ax.tick_params(labelsize=11)

        for item_index, item in enumerate(self.problem.items):

            # represent the placed items
            if item_index in self.placed_items:

                if color_items_by_profit_ratio:
                    fill_color = worst_profit_color + tuple(best_profit_color[i] - worst_profit_color[i] for i in range(
                        len(best_profit_color))) * color_interp(item_profit_ratios[item_index])
                else:
                    fill_color = (1, 0.5, 0.5)

                self.show_item(item_index, ax1, boundary_color, fill_color, container_color,
                               show_item_value_and_weight and show_value_and_weight_for_container_items, font, show_bounding_boxes, show_reference_positions)

            # determine the boundary rectangle of the outside-of-container items
            elif show_outside_items and ax2:

                outside_item_bounds[item_index] = get_bounds(
                    self.problem.items[item_index].shape)
                total_outside_item_width += abs(
                    outside_item_bounds[item_index][2] - outside_item_bounds[item_index][0])

        # show the outside-of-container items
        if show_outside_items and ax2:

            out_value_sum = 0
            out_weight_sum = 0
            row_num = max(1, int(np.log10(len(self.problem.items))
                          * (3 if len(self.problem.items) < 15 else 4)))
            row = 0
            width = 0
            max_width = 0
            row_height = 0
            height = 0
            for item_index, bounds in outside_item_bounds.items():

                out_value_sum += self.problem.items[item_index].value
                out_weight_sum += self.problem.items[item_index].weight

                if color_items_by_profit_ratio:
                    fill_color = worst_profit_color + tuple(best_profit_color[i] - worst_profit_color[i] for i in range(
                        len(best_profit_color))) * color_interp(item_profit_ratios[item_index])
                else:
                    fill_color = (1, 0.5, 0.5)

                min_x, min_y, max_x, max_y = bounds
                shape_width = abs(max_x - min_x)
                shape_height = abs(max_y - min_y)

                shape_center = get_bounding_rectangle_center(
                    self.problem.items[item_index].shape)
                position_offset = (
                    width + shape_width * 0.5 - shape_center[0], row_height + shape_height * 0.5 - shape_center[1])
                self.show_item(item_index, ax2, boundary_color, fill_color, empty_color, show_item_value_and_weight,
                               font, show_bounding_boxes, show_reference_positions, position_offset)

                height = max(height, row_height + shape_height)

                width += shape_width
                max_width += width
                if width >= total_outside_item_width / row_num:
                    row += 1
                    width = 0
                    row_height = height

            # show the value and weight outside the container
            if show_outside_value_and_weight and ax2:
                value_weight_string = "V={}".format(
                    out_value_sum if can_consider_weight else int(out_value_sum))
                if can_consider_weight:
                    value_weight_string += ", W={}".format(out_weight_sum)
                ax2.set_title("Items outside the container\n({})".format(
                    value_weight_string), fontsize=13)

        fig = plt.gcf()

        if show_plot:
            plt.show()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
            plt.close(fig)

    def show_item(self, item_index, ax, boundary_color, fill_color, container_color, show_item_value_and_weight=False, font=None, show_bounding_box=False, show_reference_position=False, position_offset=(0, 0)):
        """Show the shape of the passed item index in the indicated axis with the passed colors"""

        if item_index in self.placed_items:
            placed_shape = self.placed_items[item_index]
            shape = placed_shape.shape
        else:
            placed_shape = None
            shape = self.problem.items[item_index].shape

        x, y = get_shape_exterior_points(shape, True)
        if position_offset != (0, 0):
            x = [x_i + position_offset[0] for x_i in x]
            y = [y_i + position_offset[1] for y_i in y]

        ax.plot(x, y, color=boundary_color, linewidth=1)
        ax.fill(x, y, color=fill_color)

        if type(shape) == MultiPolygon:
            for geom in shape.geoms:
                for hole in geom.interiors:
                    x, y = get_shape_exterior_points(hole, True)
                    if position_offset != (0, 0):
                        x = [x_i + position_offset[0] for x_i in x]
                        y = [y_i + position_offset[1] for y_i in y]
                    fill_color = container_color
                    boundary_color = (0., 0., 0.)
                    ax.plot(x, y, color=boundary_color, linewidth=1)
                    ax.fill(x, y, color=fill_color)

        # show the value and weight in the centroid if required
        if show_item_value_and_weight and font:
            centroid = get_centroid(shape)
            value = self.problem.items[item_index].value
            if value / int(value) == 1:
                value = int(value)
            weight = self.problem.items[item_index].weight
            if weight / int(weight) == 1:
                weight = int(weight)
            value_weight_string = "v={}\nw={}".format(value, weight)
            item_font = dict(font)
            item_font['size'] = 9
            ax.text(centroid.x + position_offset[0], centroid.y + position_offset[1], value_weight_string,
                    horizontalalignment='center', verticalalignment='center', fontdict=item_font)

        # show the bounding box and its center if needed
        if show_bounding_box:
            bounds = get_bounds(shape)
            min_x, min_y, max_x, max_y = bounds
            x, y = (min_x, max_x, max_x, min_x,
                    min_x), (min_y, min_y, max_y, max_y, min_y)
            if position_offset != (0, 0):
                x = [x_i + position_offset[0] for x_i in x]
                y = [y_i + position_offset[1] for y_i in y]
            boundary_color = (0.5, 0.5, 0.5)
            ax.plot(x, y, color=boundary_color, linewidth=1)
            bounds_center = get_bounding_rectangle_center(shape)
            ax.plot(bounds_center[0] + position_offset[0],
                    bounds_center[1] + position_offset[1], "r.")

        # show the reference position if required
        if show_reference_position and placed_shape:
            ax.plot(placed_shape.position[0], placed_shape.position[1], "b+")

    def get_bounding_circle(self):
        """Trả về tâm và bán kính của hình tròn bao quanh nhỏ nhất chứa tất cả các hình dạng"""
        # Lấy các điểm biên của tất cả các hình
        min_x, min_y, max_x, max_y = self.get_global_bounds()
        
        # Tính tâm hình tròn
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center = (center_x, center_y)
        
        # Tính bán kính từ tâm đến điểm xa nhất
        radius = max(
            np.sqrt((max_x - center_x)**2 + (max_y - center_y)**2),
            np.sqrt((min_x - center_x)**2 + (max_y - center_y)**2),
            np.sqrt((max_x - center_x)**2 + (min_y - center_y)**2),
            np.sqrt((min_x - center_x)**2 + (min_y - center_y)**2)
        )
        
        return center, radius

    def get_bounding_circle_area(self):
        """Trả về diện tích của hình tròn bao quanh nhỏ nhất"""
        _, radius = self.get_bounding_circle()
        return np.pi * radius * radius

