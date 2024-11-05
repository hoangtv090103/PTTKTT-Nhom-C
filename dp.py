import numpy as np
from shapely.geometry import Point
from solution import Solution
from common_algo_functions import get_bounds, get_time_since
import time
import copy

# Default constants
MAX_GRID_SIZE = 20   # Maximum grid size for discretizing space
MIN_CELL_SIZE = 0.5  # Minimum cell size in the grid
ROTATION_ANGLES = [0, 90, 180, 270]  # Allowed rotation angles

def get_adaptive_grid_size(problem):
    """Calculates adaptive grid size based on item sizes."""
    min_item_dim = float('inf')
    for item in problem.items:
        bounds = get_bounds(item.shape)
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        min_item_dim = min(min_item_dim, width, height)
    
    container_bounds = get_bounds(problem.container.shape)
    container_width = container_bounds[2] - container_bounds[0]
    container_height = container_bounds[3] - container_bounds[1]
    
    # Use the smallest item dimension to set grid size
    grid_size = min(MAX_GRID_SIZE, max(5, int(min(container_width, container_height) / min_item_dim)))
    return grid_size

def discretize_space(container_shape, grid_size):
    """Discretizes the container space into a grid."""
    min_x, min_y, max_x, max_y = get_bounds(container_shape)
    x_step = (max_x - min_x) / grid_size
    y_step = (max_y - min_y) / grid_size
    
    grid_points = []
    for i in range(grid_size + 1):  # +1 to include the endpoint
        for j in range(grid_size + 1):
            x = min_x + i * x_step
            y = min_y + j * y_step
            if container_shape.contains(Point(x, y)):
                grid_points.append((x, y))
    return grid_points, x_step, y_step

def can_place_item(solution, item_index, position, rotation):
    """Kiểm tra xem có thể đặt item vào vị trí và góc quay cụ thể không"""
    return solution.add_item(item_index, position, rotation)

def get_strategic_points(container_shape, grid_points):
    """Generates a list of strategic points for placing items."""
    bounds = get_bounds(container_shape)
    strategic_points = set()
    
    # Add the corners of the container
    corners = [
        (bounds[0], bounds[1]),  # bottom-left
        (bounds[0], bounds[3]),  # top-left
        (bounds[2], bounds[1]),  # bottom-right
        (bounds[2], bounds[3])   # top-right
    ]
    for point in corners:
        if container_shape.contains(Point(point)):
            strategic_points.add(point)
    
    # Add points from the original grid
    for point in grid_points:
        if len(strategic_points) >= MAX_GRID_SIZE * 2:  # Limit the number of strategic points
            break
        strategic_points.add(point)
    
    return list(strategic_points)

def sort_items_by_value(problem):
    """Sorts items based on their value-to-weight ratio."""
    items_with_ratio = []
    for i, item in enumerate(problem.items):
        ratio = item.value / item.weight if item.weight > 0 else float('inf')
        items_with_ratio.append((i, ratio))
    return [i for i, _ in sorted(items_with_ratio, key=lambda x: x[1], reverse=True)]

def solve_problem(problem, grid_size=None, min_cell_size=MIN_CELL_SIZE,
                  rotation_angles=ROTATION_ANGLES, calculate_times=False):
    """Solves the problem using optimized dynamic programming with pruning."""
    start_time = discretize_time = dp_time = 0
    if calculate_times:
        start_time = time.time()
    
    # Use adaptive grid size
    if grid_size is None:
        grid_size = get_adaptive_grid_size(problem)
    
    # Discretize space and get strategic points
    grid_points, x_step, y_step = discretize_space(problem.container.shape, grid_size)
    strategic_points = get_strategic_points(problem.container.shape, grid_points)
    
    if calculate_times:
        discretize_time = get_time_since(start_time)
        start_time = time.time()
    
    # Sort items by value-to-weight ratio
    sorted_items = sort_items_by_value(problem)
    
    # Initialize DP table and best solution
    dp_table = {}  # {state_key: value}
    best_solution = None
    best_value = 0
    
    def dp_solve(current_solution, remaining_items, remaining_weight):
        nonlocal best_value, best_solution
        # Create a unique key for the current state
        state_key = (frozenset(current_solution.placed_items.keys()), remaining_weight)
        
        # If this state has been computed
        if state_key in dp_table:
            return dp_table[state_key]
        
        # Calculate upper bound for pruning
        remaining_values = [problem.items[item_index].value for item_index in remaining_items]
        upper_bound = current_solution.value + sum(remaining_values)
        if upper_bound <= best_value:
            return current_solution.value  # Prune this branch
        
        # Update best solution if current is better
        if current_solution.value > best_value:
            best_value = current_solution.value
            best_solution = copy.deepcopy(current_solution)
        
        # Base case: no items left or no remaining weight
        if not remaining_items or remaining_weight <= 0:
            return current_solution.value
        
        # Try placing each remaining item
        for item_index in sorted_items:
            if item_index not in remaining_items:
                continue
            
            item = problem.items[item_index]
            if item.weight > remaining_weight:
                continue
            
            # Limit placements: only try a subset of strategic points and rotations
            for pos in strategic_points:
                for angle in rotation_angles:
                    # Check if the item can be placed without modifying current_solution
                    if can_place_item(current_solution, item_index, pos, angle):
                        # Create a new solution by placing the item
                        new_solution = copy.deepcopy(current_solution)
                        new_solution.add_item(item_index, pos, angle)
                        
                        new_remaining_items = remaining_items - {item_index}
                        new_remaining_weight = remaining_weight - item.weight
                        
                        # Recursive call
                        dp_solve(new_solution, new_remaining_items, new_remaining_weight)
        
        # Store result in DP table
        dp_table[state_key] = current_solution.value
        return current_solution.value
    
    # Call dp_solve with initial state
    initial_solution = Solution(problem)
    remaining_items = set(range(len(problem.items)))
    dp_solve(initial_solution, remaining_items, problem.container.max_weight)
    
    if calculate_times:
        dp_time = get_time_since(start_time)
        # Compute total time and proportions
        total_time = discretize_time + dp_time
        time_dict = {
            "Discretizing space": (discretize_time, discretize_time / total_time),
            "Dynamic programming": (dp_time, dp_time / total_time)
        }
        return best_solution, time_dict
    
    return best_solution
