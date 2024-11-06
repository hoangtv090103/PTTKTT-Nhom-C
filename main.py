import time
import os
import logging
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, Point
from circle import Circle
from container import Container
import dp
from item import Item
from problem import Problem
from solution import Solution
import evolutionary
import greedy
from common_algo_functions import (
    get_stats,
    print_if_allowed,
    get_time_since,
    visualize_plot,
    visualize_boxplot_for_data_sequence,
    visualize_bar_plot,
    add_newlines_by_spaces,
)
from ellipse import Ellipse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("packing.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def create_knapsack_packing_problems_with_manual_solutions(can_print=False):
    logger.info("Creating knapsack packing problems with manual solutions")
    problems, solutions = [], []

    start_time = time.time()

    # Problem 1: Circle containing squares and circles
    logger.info("Creating problem 1: Circle containing squares and circles")
    container = Container(120, Circle((3.3, 3.3), radius=3.3))
    items = [
        Item(Polygon([(0, 0), (0, 4.5), (4.5, 4.5), (4.5, 0)]), 40.0, 50.0),
        Item(Circle((0, 0), 0.45), 20.0, 5.0),
        Item(Circle((0, 0), 0.45), 20.0, 10.0),
        Item(Circle((0, 0), 0.45), 20.0, 15.0),
        Item(Circle((0, 0), 0.45), 20.0, 20.0),
    ]
    problem = Problem(container, items)
    solution = Solution(problem)
    problems.append(problem)
    solutions.append(solution)

    positions_angles = [
        (0, (3.3, 3.3), 0.0),
        (1, (3.3, 6.05), 0.0),
        (2, (3.3, 0.55), 0.0),
        (3, (6.05, 3.3), 0.0),
        (4, (0.55, 3.3), 0.0),
    ]
    for item, pos, angle in positions_angles:
        print_if_allowed(solution.add_item(item, pos, angle), can_print)

    # Problem 2: Polygon with concentric shapes
    logger.info("Creating problem 2: Polygon with concentric shapes")
    container = Container(max_weight=100, shape=Point(5, 5).buffer(5, 4))
    items = [
        Item(
            MultiPolygon(
                [
                    (
                        Point(5, 5).buffer(4.7, 4).exterior.coords,
                        [tuple(Point(5, 5).buffer(4, 4).exterior.coords)],
                    )
                ]
            ),
            10.0,
            25.0,
        ),
        Item(
            MultiPolygon(
                [
                    (
                        Point(5, 5).buffer(3.7, 4).exterior.coords,
                        [tuple(Point(5, 5).buffer(3, 4).exterior.coords)],
                    )
                ]
            ),
            10.0,
            15.0,
        ),
        Item(
            MultiPolygon(
                [
                    (
                        Point(5, 5).buffer(2.7, 4).exterior.coords,
                        [tuple(Point(5, 5).buffer(2, 4).exterior.coords)],
                    )
                ]
            ),
            10.0,
            20.0,
        ),
        Item(
            MultiPolygon(
                [
                    (
                        Point(5, 5).buffer(1.7, 4).exterior.coords,
                        [tuple(Point(5, 5).buffer(1, 4).exterior.coords)],
                    )
                ]
            ),
            20.0,
            20.0,
        ),
        Item(Circle((0.0, 0.0), 0.7), 20.0, 10.0),
    ]
    problem = Problem(container, items)
    solution = Solution(problem)
    problems.append(problem)
    solutions.append(solution)

    print_if_allowed(solution.add_item(0, (5., 5.), 0.), can_print)
    print_if_allowed(solution.add_item(1, (5., 5.), 0.), can_print)
    print_if_allowed(solution.add_item(2, (5., 5.), 0.), can_print)
    print_if_allowed(solution.add_item(3, (5., 5.), 0.), can_print)
    print_if_allowed(solution.add_item(4, (5., 5.), 0.), can_print)

    # Problem 3: Rectangle with triangles and ellipses
    logger.info("Creating problem 3: Rectangle with triangles and ellipses")
    container = Container(
        max_weight=32, shape=Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
    )
    items = [
        Item(Polygon([(0, 0), (0, 6.0), (6.0, 0)]), 10.0, 20.0),
        Item(Polygon([(0, 0), (0, 6.0), (6.0, 0)]), 10.0, 10.0),
        Item(Ellipse((0, 0), 1.5, 0.3), 10.0, 5.0),
        Item(Ellipse((0, 0), 3, 0.3), 5.0, 5.0),
        Item(Ellipse((0, 0), 1.5, 0.3), 5.0, 5.0),
        Item(Ellipse((0, 0), 3, 0.3), 10.0, 5.0),
    ]
    problem = Problem(container, items)
    solution = Solution(problem)
    problems.append(problem)
    solutions.append(solution)

    positions_angles = [
        (0, (4.99, 5), 0.0),
        (1, (5.01, 5), 180.0),
        (3, (5.0, 1.65), 0.0),
        (4, (5.0, 8.35), 0.0),
    ]
    for item, pos, angle in positions_angles:
        print_if_allowed(solution.add_item(item, pos, angle), can_print)

    # Problem 4: Ellipse with mixed shapes
    logger.info("Creating problem 4: Ellipse with mixed shapes")
    container = Container(50, Ellipse((3.0, 2.0), 3.0, 2.0))
    items = [
        Item(Ellipse((0.0, 0.0), 0.7, 0.5), 5.0, 7.0),
        Item(Ellipse((0.0, 0.0), 0.3, 0.1), 7.0, 2.0),
        Item(Ellipse((0.0, 0.0), 0.2, 0.4), 8.0, 4.0),
        Item(Ellipse((0.0, 0.0), 0.5, 0.3), 3.0, 5.0),
        Item(Circle((0.0, 0.0), 0.4), 4.0, 5.0),
        Item(Circle((0.0, 0.0), 0.25), 3.0, 2.0),
        Item(Circle((0.0, 0.0), 0.2), 9.0, 5.0),
        Item(Circle((0.0, 0.0), 0.1), 4.0, 3.0),
        Item(Circle((0.0, 0.0), 0.7), 9.0, 3.0),
    ]
    problem = Problem(container, items)
    solution = Solution(problem)
    problems.append(problem)
    solutions.append(solution)

    positions_angles = [
        (0, (3.0, 1.94), 0.0),
        (2, (3.0, 3.24), 90.0),
        (3, (3.0, 2.74), 0.0),
        (4, (2.25, 3.5), 0.0),
        (5, (3.0, 3.71), 0.0),
        (6, (3.46, 3.75), 0.0),
        (7, (3.44, 3.43), 0.0),
        (8, (3.0, 0.72), 0.0),
    ]
    for item, pos, angle in positions_angles:
        print_if_allowed(solution.add_item(item, pos, angle), can_print)

    # Problem 5: Complex polygon with holes
    logger.info("Creating problem 5: Complex polygon with holes")
    container_shape = MultiPolygon(
        [
            (
                ((0, 0), (0.5, 3), (0, 5), (5, 4.5), (5, 0)),
                [
                    ((0.1, 0.1), (0.1, 0.2), (0.2, 0.2), (0.2, 0.1)),
                    ((0.3, 0.3), (0.3, 1.2), (1.6, 2.9), (0.75, 0.4)),
                    ((3.1, 1.5), (3.5, 4.5), (4.9, 4.4), (4.8, 1.2)),
                ],
            )
        ]
    )
    container = Container(max_weight=100, shape=container_shape)
    items = [
        Item(Polygon([(0, 0), (1, 1), (1, 0)]), 15.0, 32.0),
        Item(Polygon([(1, 2), (1.5, 3), (4, 5), (1, 4)]), 30.0, 100.0),
        Item(
            MultiPolygon(
                [
                    (
                        ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
                        [
                            ((0.1, 0.1), (0.1, 0.2), (0.2, 0.2), (0.2, 0.1)),
                            ((0.3, 0.3), (0.3, 0.6), (0.6, 0.6), (0.6, 0.4)),
                        ],
                    )
                ]
            ),
            12.0,
            30.0,
        ),
        Item(Polygon([(0.1, 0.1), (0.1, 0.2), (0.2, 0.2)]), 10.0, 10.0),
        Item(
            MultiPolygon(
                [
                    (
                        ((0.0, 0.0), (0.0, 1.4), (2.0, 1.3), (2.0, 0.0)),
                        [
                            ((0.1, 0.1), (0.1, 0.15), (0.15, 0.15), (0.15, 0.1)),
                            ((0.2, 0.2), (0.2, 1.2), (1.8, 1.1), (1.8, 0.2)),
                        ],
                    )
                ]
            ),
            1.0,
            5.0,
        ),
        Item(Circle((0.0, 0.0), 0.4), 1.0, 14.0),
        Item(Circle((0.0, 0.0), 0.1), 2.0, 12.0),
        Item(Ellipse((0.0, 0.0), 0.5, 0.2), 3.0, 12.0),
        Item(Polygon([(0.0, 0.0), (0.0, 0.3), (0.3, 0.3)]), 1.0, 10.0),
        Item(Ellipse((0.0, 0.0), 0.8, 0.3), 10.0, 12.0),
        Item(Ellipse((0.0, 0.0), 0.1, 0.05), 1.0, 2.0),
        # random items
        # Item(shape_functions.create_random_polygon(0, 0, 0.8, 0.8, 10), 1., 5.),
        # Item(shape_functions.create_random_triangle_in_rectangle_corner(0, 0, 0.8, 0.8), 1., 5.),
        # Item(shape_functions.create_random_quadrilateral_in_rectangle_corners(0, 0, 0.8, 0.8), 1., 5.),
        # out-items
        Item(Circle((0.0, 0.0), 0.2), 50.0, 1.0),
    ]
    problem = Problem(container, items)
    solution = Solution(problem)
    problems.append(problem)
    solutions.append(solution)

    print_if_allowed(solution.add_item(0, (1.2, 0.5), 0.), can_print)
    print_if_allowed(solution.add_item(1, (2., 3.), 0.), can_print)
    print_if_allowed(solution.add_item(2, (2.5, 2.5), 0.), can_print)
    print_if_allowed(solution.move_item_in_direction(2, (1, -1), evolutionary.MUTATION_MODIFY_MOVE_UNTIL_INTERSECTION_POINT_NUM,
                     evolutionary.MUTATION_MODIFY_MOVE_UNTIL_INTERSECTION_MIN_DIST_PROPORTION, 9999), can_print)
    print_if_allowed(solution.add_item(3, (2.5, 2.4), 0.), can_print)
    print_if_allowed(solution.add_item(4, (3., 0.7), 0.), can_print)
    print_if_allowed(solution.add_item(5, (3.03, 0.73), 0.), can_print)
    print_if_allowed(solution.add_item(6, (3.45, 1.02), 0.), can_print)
    print_if_allowed(solution.add_item(7, (3., 3.82), 45.), can_print)
    print_if_allowed(solution.add_item(8, (2.4, 0.7), 0.), can_print)
    print_if_allowed(solution.move_item(0, (0.29, 0)), can_print)
    # print_if_allowed(solution.move_item_to(0, (1.49, 2.5)), can_print)
    print_if_allowed(solution.rotate_item_to(0, 180.), can_print)
    print_if_allowed(solution.rotate_item(0, 90.), can_print)
    # print_if_allowed(solution.remove_item(0), can_print)
    # print_if_allowed(solution.rotate_item(4, 20), can_print)
    # print_if_allowed(solution.move_item(7, (1, 0)), can_print)
    # print_if_allowed(solution.rotate_item(7, -45), can_print)
    # print_if_allowed(solution.move_item(5, (-0.4, 0)), can_print)
    print_if_allowed(solution.add_item(9, (1.2, 4.07), 15.), can_print)
    print_if_allowed(solution.add_item(10, (3.6, 0.45), 30.), can_print)
    # print_if_allowed(solution.add_item(11, (4.5, 0.5), 0.), can_print)
    # Problem 6: Complex polygon with multiple holes
    logger.info("Creating problem 6: Complex polygon with multiple holes")
    container = Container(
        max_weight=150,
        shape=MultiPolygon(
            [
                (
                    ((0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)),
                    [
                        ((0.7, 0.7), (1.5, 0.7), (1.5, 1.5), (0.7, 1.5)),
                        ((2.4, 0.3), (4.3, 0.3), (4.3, 4.3), (2.4, 4.3)),
                        ((0.7, 2.7), (1.5, 2.7), (1.5, 3.5), (0.7, 3.5)),
                    ],
                )
            ]
        ),
    )
    items = [
        Item(Polygon([(0.0, 0.0), (1.6, 0.0), (1.4, 0.2), (1.7, 1.0)]), 6.0, 13.0),
        Item(
            Polygon([(0.0, 0.0), (1.6, 3.0), (2.8, 2.9), (1.5, 2.7), (1.9, 1.6)]),
            11.0,
            12.0,
        ),
        Item(Polygon([(0.0, 0.0), (1.8, 1.5), (0.0, 2.8)]), 15.0, 25.0),
        Item(Polygon([(0.0, 0.0), (1.5, 0.0), (1.5, 0.2), (0.0, 0.2)]), 14.0, 10.0),
        Item(Polygon([(0.0, 0.0), (2.5, 0.0), (1.5, 0.2), (0.0, 0.2)]), 10.0, 12.0),
        Item(
            Polygon([(0.0, 0.0), (1.6, 0.0), (0.8, 0.45), (0.6, 0.7), (0.0, 0.45)]),
            17.0,
            8.0,
        ),
        Item(Polygon([(0.0, 0.0), (1.5, 0.0), (0.8, 0.15), (0.0, 0.1)]), 13.0, 12.0),
        Item(Polygon([(0.0, 0.0), (1.5, 0.0), (0.8, 0.15), (0.0, 0.1)]), 15.0, 7.0),
        Item(Ellipse((0.0, 0.0), 0.5, 0.3), 15.0, 8.0),
        Item(Ellipse((0.0, 0.0), 0.2, 0.8), 14.0, 21.0),
        Item(Circle((0.0, 0.0), 0.2), 18.0, 18.0),
        Item(Circle((0.0, 0.0), 0.6), 11.0, 12.0),
        Item(Circle((0.0, 0.0), 0.35), 12.0, 9.0),
    ]
    problem = Problem(container, items)
    solution = Solution(problem)
    problems.append(problem)
    solutions.append(solution)

    positions_angles = [
        (0, (0.9, 2.02), 0.0),
        (3, (0.78, 0.12), 0.0),
        (4, (2.8, 0.12), 0.0),
        (5, (0.8, 3.85), 0.0),
        (6, (0.78, 0.3), 0.0),
        (7, (2.3, 2.57), 90.0),
        (8, (0.3, 2.98), 90.0),
        (9, (2.17, 1.05), 0.0),
        (10, (1.8, 0.45), 0.0),
        (11, (1.77, 4.38), 0.0),
        (12, (0.35, 4.63), 0.0),
    ]
    for item, pos, angle in positions_angles:
        print_if_allowed(solution.add_item(item, pos, angle), can_print)

    # Problem 7: Complex polygon with irregular shape
    logger.info("Creating problem 7: Complex polygon with irregular shape")
    container = Container(
        122,
        Polygon(
            [
                (3.5, 0.6),
                (0.5, 0.9),
                (3.7, 5.5),
                (1.7, 4.0),
                (0.0, 6.5),
                (0.2, 8.6),
                (0.8, 9.8),
                (1.7, 8.9),
                (2, 9.1),
                (4.4, 9.3),
                (4.2, 6.7),
                (4.9, 7.5),
                (6.5, 8.4),
                (6.6, 7.9),
                (7.4, 8.2),
                (8.7, 5.5),
                (9.3, 4.8),
                (6.3, 0.2),
                (5.0, 3.5),
                (5, 0.7),
            ]
        ),
    )
    items = [Item(Polygon([(0, 3), (0, 2.), (4., 0)]), 5., 6.),
            Item(Polygon([(0, 0), (1., 2.), (2.5, 2), (1, 1.2)]), 10., 7.),
            Item(Polygon([(0, 1), (1, 2.), (3., 0)]), 9., 4.),
            Item(Polygon([(0, 0.5), (1, 1.), (3, 1), (2., 0)]), 19., 14.),
            Item(Polygon([(0, 0.6), (2, 1), (2., 1.5), (1.2, 1.5)]), 19., 15.),
            Item(Polygon([(0, 0), (0, 2.), (0.5, 2), (0.5, 0.5), (2.5, 0.5), (2.5, 0)]), 7., 15.),
            Item(MultiPolygon([(((0.0, 0.0), (0.0, 1.8), (1.0, 2.7), (2.3, 0.0)),
                                [((0.2, 0.2), (0.2, 1.4), (0.7, 2.1), (1.8, 0.5))])]), 12., 6.),
            Item(MultiPolygon([(((0.0, 0.0), (1.0, 1.8), (2.0, 2.5), (2.6, 0.7)),
                                [((0.2, 0.2), (1.2, 1.4), (2.1, 1.7))])]), 7., 13.),
            Item(Ellipse((0, 0), 0.5, 0.2), 4., 9.),
            Item(Ellipse((0, 0), 0.2, 1.5), 21., 14.),
            Item(Ellipse((0, 0), 2.5, 3.5), 16., 30.),
            Item(Circle((0, 0), 0.4), 7., 12.),
            Item(Circle((0, 0), 0.3), 10., 3.),
            Item(Circle((0, 0), 1.), 1., 3.)]
    problem = Problem(container, items)
    solution = Solution(problem)
    problems.append(problem)
    solutions.append(solution)

    print_if_allowed(solution.add_item(0, (5.73, 3.02), 318.), can_print)
    print_if_allowed(solution.add_item(1, (6.3, 4.1), 40.), can_print)
    print_if_allowed(solution.add_item(2, (4.58, 2.5), 315.), can_print)
    print_if_allowed(solution.add_item(3, (1.3, 5.4), 320.), can_print)
    print_if_allowed(solution.add_item(4, (1.4, 1.7), 20.), can_print)
    print_if_allowed(solution.add_item(5, (2.9, 7.9), 180.), can_print)
    print_if_allowed(solution.add_item(6, (8.2, 4), 300.), can_print)
    print_if_allowed(solution.add_item(7, (2.5, 7.4), 340.), can_print)
    print_if_allowed(solution.add_item(8, (7.3, 4.), 320.), can_print)
    print_if_allowed(solution.add_item(9, (2.9, 3.9), 330.), can_print)
    print_if_allowed(solution.add_item(11, (7.8, 4.4), 0.), can_print)
    print_if_allowed(solution.add_item(13, (6.2, 6.8), 0.), can_print)

    # Problem 8: Rectangle with special shape
    logger.info("Creating problem 8: Rectangle with special shape")
    container = Container(
        100,
        Polygon([(0.0, 0.0), (0.0, 5.0), (2.5, 3.4), (5.0, 5.0), (5.0, 0), (2.5, 1.6)]),
    )
    items = [Item(Polygon([(0., 0.), (0., 3.), (0.25, 3.), (0.25, 0.25), (2., 2.5), (3.75, 0.25), (3.75, 3.), (4., 3.), (4., 0.), (3.75, 0.), (2., 2.), (0.25, 0.)]), 100., 100.),
             Item(Polygon([(0., 0.), (1.6, 1.), (1.8, 1.9), (0.9, 1.6)]), 11., 12.),
             Item(Polygon([(0., 0.), (1.8, 2.5), (0., 1.8)]), 15., 5.),
             Item(Polygon([(0., 0.), (0.5, 0.), (1.2, 0.4), (0., 0.5)]), 4., 10.),
             Item(Polygon([(0., 0.), (2.5, 0.), (1.5, 0.2), (0., 0.5)]), 1., 2.),
             Item(Polygon([(0., 0.), (0.7, 0.25), (1.6, 1.5), (0.6, 0.7), (0., 0.45)]), 17., 8.),
             Item(Polygon([(0., 0.), (0.8, 0.5), (1.5, 1.2), (0., 0.5)]), 13., 11.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.2, 0.6), (0., 0.3)]), 15., 7.),
             Item(Ellipse((0., 0.), 0.6, 0.4), 15., 8.),
             Item(Ellipse((0., 0.), 2., 0.5), 15., 8.),
             Item(Ellipse((0., 0.), 0.5, 0.3), 24., 6.),
             Item(Ellipse((0., 0.), 0.4, 0.1), 4., 3.),
             Item(Circle((0., 0.), 0.6), 11., 2.),
             Item(Circle((0., 0.), 0.35), 12., 4.),
             Item(Circle((0., 0.), 0.2), 18., 8.)]
    problem = Problem(container, items)
    problems.append(problem)

    solution = Solution(problem)
    solutions.append(solution)

    print_if_allowed(solution.add_item(0, (2.5, 2.02), 0.), can_print)

    # Problem 9: Circle with concentric rings
    logger.info("Creating problem 9: Circle with concentric rings")
    container = Container(200, Point(5, 5).buffer(5, 3))
    items = [Item(MultiPolygon([(Point(5, 5).buffer(4.7, 2).exterior.coords,
                                [((9., 5.), (5., 1.), (1., 5.), (5., 9.))])]), 120., 110.),
            Item(Polygon([(0., 0.), (0., 5.), (5., 5.), (5., 0.)]), 50., 80.),
            Item(Polygon([(1., 4.2), (1.5, 2.), (4., 0)]), 15., 14.),
            Item(Polygon([(0, 0), (1., 2.), (2.5, 2), (1, 1.2)]), 11., 11.),
            Item(Polygon([(0, 1), (1, 2.), (3., 0)]), 11., 4.),
            Item(Polygon([(0, 0.5), (1, 1.), (3, 1), (2., 0)]), 19., 14.),
            Item(Polygon([(0, 0.4), (1.8, .8), (1.5, 1.3), (1.2, 3.3)]), 17., 15.),
            Item(Polygon([(0, 0), (0, 2.), (0.9, 2), (0.9, 0.5), (1.5, 0.5), (1.5, 0)]), 70., 15.),
            Item(Ellipse((0, 0), 0.8, 1.2), 14., 13.),
            Item(Ellipse((0, 0), 1.2, 1.5), 12., 6.),
            Item(Ellipse((0, 0), 2.5, 1.7), 16., 10.),
            Item(Circle((0, 0), 0.7), 17., 11.),
            Item(Circle((0, 0), 0.8), 13., 10.),
            Item(Circle((0, 0), 1.), 4., 4.),
            Item(Circle((0, 0), 2.), 22., 8.)]
    problem = Problem(container, items)
    problems.append(problem)

    solution = Solution(problem)
    solutions.append(solution)

    print_if_allowed(solution.add_item(0, (5., 5.), 0.), can_print)
    print_if_allowed(solution.add_item(1, (5., 5.), 45.), can_print)

    # Problem 10: Cross-shaped container
    logger.info("Creating problem 10: Cross-shaped container")
    container = Container(
        150,
        Polygon(
            [
                (2.0, 5.0),
                (3.0, 5),
                (3.0, 3.0),
                (5.0, 3.0),
                (5.0, 2.0),
                (3.0, 2.0),
                (3.0, 0.0),
                (2.0, 0.0),
                (2.0, 2.0),
                (0.0, 2.0),
                (0.0, 3.0),
                (2.0, 3.0),
            ]
        ),
    )
    items = [Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95)]), 10., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95)]), 10., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95)]), 10., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95)]), 10., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95)]), 10., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95)]), 10., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95)]), 10., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95)]), 10., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95)]), 10., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95)]), 10., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95), (0., 0.95)]), 20., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95), (0., 0.95)]), 20., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95), (0., 0.95)]), 20., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95), (0., 0.95)]), 20., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95), (0., 0.95)]), 20., 10.),
             Item(Polygon([(0., 0.), (1.5, 0.), (1.5, 0.95), (0., 0.95)]), 20., 10.),
             Item(Polygon([(0., 0.), (0.8, 0.), (0.8, 0.45), (0., 0.45)]), 20., 30.),
             Item(Polygon([(0., 0.), (0.8, 0.), (0.8, 0.45), (0., 0.45)]), 20., 30.),
             Item(Polygon([(0., 0.), (0.8, 0.), (0.8, 0.1), (0., 0.1)]), 5., 25.),
             Item(Polygon([(0., 0.), (0.8, 0.), (0.8, 0.1), (0., 0.1)]), 5., 25.)]
    problem = Problem(container, items)
    problems.append(problem)

    solution = Solution(problem)
    solutions.append(solution)

    print_if_allowed(solution.add_item(0, (4.23, 2.48), 0.), can_print)
    print_if_allowed(solution.add_item(1, (4.23, 2.52), 180.), can_print)
    print_if_allowed(solution.add_item(2, (0.77, 2.48), 0.), can_print)
    print_if_allowed(solution.add_item(3, (0.77, 2.52), 180.), can_print)
    print_if_allowed(solution.add_item(4, (2.48, 0.76), 270.), can_print)
    print_if_allowed(solution.add_item(5, (2.52, 0.76), 90.), can_print)
    print_if_allowed(solution.add_item(6, (2.48, 4.24), 270.), can_print)
    print_if_allowed(solution.add_item(7, (2.52, 4.24), 90.), can_print)
    print_if_allowed(solution.add_item(8, (2.5, 2.48), 0.), can_print)
    print_if_allowed(solution.add_item(9, (2.5, 2.52), 180.), can_print)
    print_if_allowed(solution.add_item(16, (2.5, 3.25), 0.), can_print)
    print_if_allowed(solution.add_item(17, (2.5, 1.75), 0.), can_print)
    print_if_allowed(solution.add_item(18, (1.64, 2.5), 90.), can_print)
    print_if_allowed(solution.add_item(19, (3.36, 2.5), 90.), can_print)

    return problems, [str(i + 1) for i in range(len(problems))], solutions


def execute_algorithm_with_params(params):
    algorithm, problem, calculate_times, calculate_value_evolution = params[:4]
    logger.info(f"Executing algorithm {algorithm.__name__} on problem")
    start_time = time.time()

    if calculate_value_evolution:
        # Nếu hàm solve_problem không hỗ trợ return_population_fitness_per_generation, bỏ qua tham số này
        result = algorithm(problem, calculate_times=calculate_times)
    else:
        result = algorithm(problem, calculate_times=calculate_times)

    elapsed_time = get_time_since(start_time)
    solution = result[0]
    value_evolution = result[1] if calculate_value_evolution else None

    logger.info(
        f"Algorithm {algorithm.__name__} completed in {elapsed_time}ms with solution value {solution.value}"
    )
    return solution, solution.value, value_evolution, elapsed_time, None


def execute_algorithm(
    algorithm,
    problem,
    execution_num=1,
    calculate_times=False,
    calculate_value_evolution=False,
):
    logger.info(
        f"Starting {execution_num} executions of algorithm {algorithm.__name__}"
    )
    param_tuples = [
        (algorithm, problem, calculate_times, calculate_value_evolution)
        for _ in range(execution_num)
    ]
    solutions = []
    values = []
    value_evolutions = []
    times = []
    time_divisions = []

    for i, params in enumerate(param_tuples):
        logger.info(f"Starting execution {i+1}/{execution_num}")
        solution, value, value_evolution, elapsed_time, time_division = (
            execute_algorithm_with_params(params)
        )
        solutions.append(solution)
        values.append(value)
        value_evolutions.append(value_evolution)
        times.append(elapsed_time)
        time_divisions.append(time_division)
        logger.info(f"Execution {i+1} completed with value {value}")

    logger.info(f"All {execution_num} executions completed")
    return solutions, values, value_evolutions, times, time_divisions


def run():
    logger.info("Starting experiment run")
    problems, problem_names, manual_solutions = (
        create_knapsack_packing_problems_with_manual_solutions()
    )

    experiment_dict = {}

    for problem, problem_name, solution in zip(
        problems, problem_names, manual_solutions
    ):
        logger.info(f"Processing problem {problem_name}")
        experiment_dict[problem_name] = {
            "problem": problem,
            "manual_solution": solution,
            "algorithms": {},
        }

        for algorithm_name, algorithm in [
            ("Tham lam", greedy.solve_problem),
            ("Quy hoạch động", dp.solve_problem),
            ("Di truyền", evolutionary.solve_problem),
        ]:
            logger.info(f"Running algorithm {algorithm_name} on problem {problem_name}")
            solutions, values, value_evolutions, times, time_divisions = (
                execute_algorithm(
                    algorithm=algorithm,
                    problem=problem,
                    execution_num=10,
                    calculate_times=True,
                    calculate_value_evolution=True,
                )
            )
            experiment_dict[problem_name]["algorithms"][algorithm_name] = {
                "solutions": solutions,
                "values": values,
                "value_evolutions": value_evolutions,
                "times": times,
                "time_divisions": time_divisions,
            }
            logger.info(
                f"Completed algorithm {algorithm_name} on problem {problem_name}"
            )

    logger.info("Experiment run completed")
    return experiment_dict


def visualize_and_save_experiments(
    experiment_dict,
    output_dir,
    can_plots_show_value_and_weight=True,
    show_problem_stats=False,
    save_problem_stats=True,
    show_manual_solution_plots=False,
    save_manual_solution_plots=True,
    show_algorithm_solution_plots=False,
    save_algorithm_solution_plots=True,
    show_value_evolution_plots=False,
    save_value_evolution_plots=True,
    show_time_division_plots=False,
    save_time_division_plots=True,
    show_algorithm_comparison=False,
    save_algorithm_comparison=True,
    show_aggregated_result_tables=True,
    save_aggregated_result_tables=True,
):
    """Show and/or save different plots of the results of the experiments represented in the passed dictionary, as specified by the parameters"""
    logger.info("Starting visualization and saving of experiment results")

    # if needed, save statistics about the problems and their manual solutions
    if show_problem_stats or save_problem_stats:
        logger.info("Processing problem statistics")
        problem_names = list(experiment_dict.keys())
        fields = [
            "Item num.",
            "Opt. % item num. in cont.",
            "Opt. % item value in cont.",
            "Item weight % of max weight",
            "Item area % of max area",
            "Cont. weight satur. %",
            "Cont. area satur. %",
        ]

        data_frame = pd.DataFrame(index=problem_names, columns=fields)

        for problem_name in experiment_dict.keys():
            problem = experiment_dict[problem_name]["problem"]
            solution = experiment_dict[problem_name]["manual_solution"]
            if type(solution) == Solution:
                problem_results = [
                    len(problem.items),
                    round(len(solution.placed_items) / len(problem.items) * 100, 2),
                    round(
                        sum(
                            [
                                problem.items[item_index].value
                                for item_index in solution.placed_items.keys()
                            ]
                        )
                        / sum([item.value for item in problem.items])
                        * 100,
                        2,
                    ),
                    round(
                        sum([item.weight for item in problem.items])
                        / problem.container.max_weight
                        * 100,
                        2,
                    ),
                    round(
                        sum([item.shape.area for item in problem.items])
                        / problem.container.shape.area
                        * 100,
                        2,
                    ),
                    round(
                        sum(
                            [
                                problem.items[item_index].weight
                                for item_index in solution.placed_items.keys()
                            ]
                        )
                        / problem.container.max_weight
                        * 100,
                        2,
                    ),
                    round(
                        sum(
                            [
                                problem.items[item_index].shape.area
                                for item_index in solution.placed_items.keys()
                            ]
                        )
                        / problem.container.shape.area
                        * 100,
                        2,
                    ),
                ]
                data_frame.loc[problem_name] = problem_results

        if len(data_frame) > 0:
            data_frame.index = [
                ("Problem " + name if len(name) < 5 else name)
                for name in data_frame.index
            ]
            min_row = data_frame.min()
            max_row = data_frame.max()
            mean_row = data_frame.mean()
            std_row = data_frame.std()
            data_frame.loc["Min"] = round(min_row, 2)
            data_frame.loc["Max"] = round(max_row, 2)
            data_frame.loc["Mean"] = round(mean_row, 2)
            data_frame.loc["Std"] = round(std_row, 2)
            if (max_row != min_row).all():
                data_frame.loc["Std / (max - min) %"] = round(
                    std_row / (max_row - min_row) * 100, 2
                )
            else:
                data_frame.loc["Std / (max - min) %"] = float("inf")

            if show_problem_stats:
                print(data_frame.to_string())

            if save_problem_stats:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                data_frame.to_excel(output_dir + "problem_stats.xlsx")
                data_frame.to_latex(output_dir + "problem_stats.tex")
                logger.info("Saved problem statistics")

    # create the problem results directory (if not done yet)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # for each problem's results, show or save plots if needed
    if (
        show_manual_solution_plots
        or save_manual_solution_plots
        or show_algorithm_solution_plots
        or save_algorithm_solution_plots
        or show_algorithm_comparison
        or save_algorithm_comparison
    ):
        for problem_name in experiment_dict.keys():
            logger.info(f"Processing visualizations for problem {problem_name}")

            problem, manual_solution, algorithm_dict = experiment_dict[
                problem_name
            ].values()

            # create a subdirectory to store the solutions of the problem (if not done yet)
            problem_dir_path = output_dir + problem_name + "/"
            if not os.path.exists(problem_dir_path):
                os.mkdir(problem_dir_path)
                logger.info(f"Created problem directory: {problem_dir_path}")

            plotted_problem_name = (
                "Problem " + problem_name if len(problem_name) < 5 else problem_name
            )

            # if needed, show/save a plot of the initial state (empty solution), and the final state of a manual solution
            if show_manual_solution_plots or save_manual_solution_plots:
                logger.info(
                    f"Processing manual solution plots for problem {problem_name}"
                )
                empty_solution = Solution(problem)

                empty_solution.visualize(
                    title_override=plotted_problem_name + " - Initial state",
                    show_plot=show_manual_solution_plots,
                    save_path=problem_dir_path + "empty_solution.png"
                    if save_manual_solution_plots
                    else None,
                    show_item_value_and_weight=can_plots_show_value_and_weight,
                    show_value_weight_ratio_bar=can_plots_show_value_and_weight,
                )

                if type(manual_solution) == Solution:
                    manual_solution.visualize(
                        title_override=plotted_problem_name + " - Manual solution",
                        show_plot=show_manual_solution_plots,
                        save_path=problem_dir_path + "manual_solution.png"
                        if save_manual_solution_plots
                        else None,
                        show_item_value_and_weight=can_plots_show_value_and_weight,
                        show_value_weight_ratio_bar=can_plots_show_value_and_weight,
                    )

            # if required, show/save plots of the solutions of each algorithm
            if show_algorithm_solution_plots or save_algorithm_solution_plots:
                for algorithm_name, subdict in algorithm_dict.items():
                    logger.info(
                        f"Processing solution plots for algorithm {algorithm_name}"
                    )
                    for i, solution in enumerate(subdict["solutions"]):
                        solution.visualize(
                            title_override=plotted_problem_name
                            + " - "
                            + algorithm_name
                            + " solution",
                            show_plot=show_algorithm_solution_plots,
                            save_path=problem_dir_path
                            + ""
                            + algorithm_name.lower()
                            + "_exec"
                            + str(i + 1)
                            + "_solution.png"
                            if save_algorithm_solution_plots
                            else None,
                            show_item_value_and_weight=can_plots_show_value_and_weight,
                            show_value_weight_ratio_bar=can_plots_show_value_and_weight,
                        )

            # if required, show/save plots of the value evolution of each algorithm
            if show_value_evolution_plots or save_value_evolution_plots:
                for algorithm_name, subdict in algorithm_dict.items():
                    logger.info(
                        f"Processing value evolution plots for algorithm {algorithm_name}"
                    )
                    for i, value_evolution in enumerate(subdict["value_evolutions"]):
                        if value_evolution:
                            if (
                                type(value_evolution) == list
                                and type(value_evolution[0]) == list
                            ):
                                visualize_boxplot_for_data_sequence(
                                    data_lists=value_evolution,
                                    title=plotted_problem_name
                                    + " - Population fitness per generation",
                                    show_plot=show_value_evolution_plots,
                                    save_path=problem_dir_path
                                    + ""
                                    + algorithm_name.lower()
                                    + "_exec"
                                    + str(i + 1)
                                    + "_fitness_evolution.png"
                                    if save_value_evolution_plots
                                    else None,
                                )
                            else:
                                visualize_plot(
                                    values=value_evolution,
                                    title=plotted_problem_name
                                    + " - "
                                    + algorithm_name
                                    + " solution value per iteration",
                                    show_plot=show_value_evolution_plots,
                                    save_path=problem_dir_path
                                    + ""
                                    + algorithm_name.lower()
                                    + "_exec"
                                    + str(i + 1)
                                    + "_value_evolution.png"
                                    if save_value_evolution_plots
                                    else None,
                                )

            # if required, show/save plots of the time division in tasks of each algorithm
            if show_time_division_plots or save_time_division_plots:
                for algorithm_name, subdict in algorithm_dict.items():
                    logger.info(
                        f"Processing time division plots for algorithm {algorithm_name}"
                    )
                    for i, time_division in enumerate(subdict["time_divisions"]):
                        if (
                            time_division and time_division.values()
                        ):  # Ensure time_division is not None and has values
                            visualize_bar_plot(
                                values=[
                                    value_pair[0]
                                    for value_pair in time_division.values()
                                ],
                                labels=[
                                    add_newlines_by_spaces(label, 7)
                                    for label in list(time_division.keys())
                                ],
                                title="Problem "
                                + problem_name
                                + " - "
                                + algorithm_name
                                + " time per task (milliseconds)",
                                show_plot=show_algorithm_solution_plots,
                                save_path=problem_dir_path
                                + ""
                                + algorithm_name.lower()
                                + "_exec"
                                + str(i + 1)
                                + "_time_division.png"
                                if save_algorithm_solution_plots
                                else None,
                            )

            # if needed, show/save plots that compare the value and time of each algorithm considering multiple executions
            if show_algorithm_comparison or save_algorithm_comparison:
                logger.info(
                    f"Processing algorithm comparison plots for problem {problem_name}"
                )
                visualize_boxplot_for_data_sequence(
                    data_lists=[
                        experiment_dict[problem_name]["algorithms"][algo_name]["values"]
                        for algo_name in experiment_dict[problem_name][
                            "algorithms"
                        ].keys()
                    ],
                    title="Problem " + problem_name + " - Solution value by algorithm",
                    labels=experiment_dict[problem_name]["algorithms"].keys(),
                    show_plot=show_algorithm_comparison,
                    save_path=problem_dir_path + "value_comparison.png"
                    if save_algorithm_comparison
                    else None,
                )
                visualize_boxplot_for_data_sequence(
                    data_lists=[
                        experiment_dict[problem_name]["algorithms"][algo_name]["times"]
                        for algo_name in experiment_dict[problem_name][
                            "algorithms"
                        ].keys()
                    ],
                    title="Problem "
                    + problem_name
                    + " - Computational time (milliseconds) by algorithm",
                    labels=experiment_dict[problem_name]["algorithms"].keys(),
                    y_scale_override="log",
                    show_plot=show_algorithm_comparison,
                    save_path=problem_dir_path + "time_comparison.png"
                    if save_algorithm_comparison
                    else None,
                )

    # if needed, save tables with an aggregation of the value and time results of the executions of each problem (or just show them)
    if show_aggregated_result_tables or save_aggregated_result_tables:
        logger.info("Processing aggregated result tables")
        problem_names = list(experiment_dict.keys())
        algorithm_names = [
            algo_name
            for algo_name in experiment_dict[problem_names[0]]["algorithms"].keys()
        ]
        fields = ["mean", "std", "min", "med", "max"]
        for concept in ["value", "time"]:
            algo_field_tuples = [
                (algo_name, field) for algo_name in algorithm_names for field in fields
            ]
            if concept == "value":
                algo_field_tuples += [("Manual", "optim.")]
            multi_index = pd.MultiIndex.from_tuples(
                algo_field_tuples, names=["Algorithm", "Statistic"]
            )
            data_frame = pd.DataFrame(index=problem_names, columns=multi_index)
            for problem_name in experiment_dict.keys():
                problem_results = list()
                for algo_name in algorithm_names:
                    mean, std, min_, median, max_ = get_stats(
                        experiment_dict[problem_name]["algorithms"][algo_name][
                            concept + "s"
                        ],
                        2 if concept == "value" else 0,
                    )
                    problem_results.extend([mean, std, min_, median, max_])
                if concept == "value":
                    if (
                        type(experiment_dict[problem_name]["manual_solution"])
                        == Solution
                    ):
                        problem_results.append(
                            experiment_dict[problem_name]["manual_solution"].value
                        )
                    else:
                        problem_results.append(
                            experiment_dict[problem_name]["manual_solution"]
                        )
                data_frame.loc[problem_name] = problem_results
            data_frame.index = [
                ("Problem " + name if len(name) < 5 else name)
                for name in data_frame.index
            ]
            if show_aggregated_result_tables:
                print(
                    "{} results:\n{}\n".format(
                        concept.capitalize(), data_frame.to_string()
                    )
                )
            if save_aggregated_result_tables:
                data_frame.to_excel(output_dir + concept + "_results.xlsx")
                data_frame.to_latex(output_dir + concept + "_results.tex")
                logger.info(f"Saved {concept} results")

    logger.info("Completed visualization and saving of experiment results")


if __name__ == "__main__":
    logger.info("Starting main program execution")
    experiment_dict = run()
    visualize_and_save_experiments(experiment_dict, output_dir="experiment_results/")
    logger.info("Program execution completed")
