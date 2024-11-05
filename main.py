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
from common_algo_functions import get_stats, print_if_allowed, get_time_since, visualize_plot, visualize_boxplot_for_data_sequence, visualize_bar_plot, add_newlines_by_spaces
from ellipse import Ellipse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('packing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_knapsack_packing_problems_with_manual_solutions(can_print=False):
    logger.info("Creating knapsack packing problems with manual solutions")
    problems, solutions = [], []

    # Vấn đề 1: Hình tròn chứa các hình vuông, tròn và ellipse
    logger.info("Creating problem 1: Circle containing squares, circles and ellipses")
    container = Container(120, Circle((3.3, 3.3), radius=3.3))
    items = [
        Item(Polygon([(0, 0), (0, 4.5), (4.5, 4.5), (4.5, 0)]), 40., 50.),
        Item(Circle((0, 0), 0.45), 20, 5),
        Item(Ellipse((0, 0), 0.45, 0.3), 20, 10),
        Item(Circle((0, 0), 0.45), 20, 15),
        Item(Ellipse((0, 0), 0.3, 0.45), 20, 20)
    ]
    problem = Problem(container, items)
    solution = Solution(problem)
    problems.append(problem)
    solutions.append(solution)
    positions_angles = [(0, (3.3, 3.3), 0.), (1, (3.3, 6.05), 0.), (2, (3.3, 0.55), 45.), 
                       (3, (6.05, 3.3), 0.), (4, (0.55, 3.3), 90.)]
    for item, pos, angle in positions_angles:
        print_if_allowed(solution.add_item(item, pos, angle), can_print)

    # Vấn đề 2: Đa giác chứa các ellipse đồng tâm
    logger.info("Creating problem 2: Polygon containing concentric ellipses") 
    container = Container(100., Point(5, 5).buffer(5, 4))
    items = [Item(Ellipse((5, 5), i, i-0.2), 10, 25-i*5) for i in range(1, 6)]
    problem = Problem(container, items)
    solution = Solution(problem)
    problems.append(problem)
    solutions.append(solution)
    for i in range(5):
        print_if_allowed(solution.add_item(i, (5, 5), 0.), can_print)

    # Vấn đề 3: Hình chữ nhật chứa các ellipse khác kích thước
    logger.info("Creating problem 3: Rectangle containing different sized ellipses")
    container = Container(150, Polygon([(0,0), (10,0), (10,8), (0,8)]))
    items = [Item(Ellipse((0,0), 1+0.3*i, 0.8+0.2*i), 15+i*5, 20+i*10) for i in range(5)]
    problem = Problem(container, items)
    solution = Solution(problem)
    problems.append(problem)
    solutions.append(solution)
    positions = [(2,4), (4,4), (6,4), (8,4), (8,6)]
    angles = [0, 45, 90, 135, 180]
    for i, (pos, angle) in enumerate(zip(positions, angles)):
        print_if_allowed(solution.add_item(i, pos, angle), can_print)

    # Vấn đề 4: Hình ellipse lớn chứa các hình tròn nhỏ
    logger.info("Creating problem 4: Large ellipse containing small circles")
    container = Container(200, Ellipse((5,5), 6, 4))
    items = [Item(Circle((0,0), 1), 30, 40) for _ in range(5)]
    problem = Problem(container, items)
    solution = Solution(problem)
    problems.append(problem)
    solutions.append(solution)
    positions = [(3,5), (5,3), (7,5), (5,7), (5,5)]
    for i, pos in enumerate(positions):
        print_if_allowed(solution.add_item(i, pos, 0), can_print)

    # Vấn đề 5: Hình vuông chứa hỗn hợp ellipse và hình vuông
    logger.info("Creating problem 5: Square containing mix of ellipses and squares")
    container = Container(180, Polygon([(0,0), (8,0), (8,8), (0,8)]))
    items = [
        Item(Ellipse((0,0), 1, 0.5), 25, 35),
        Item(Polygon([(0,0), (2,0), (2,2), (0,2)]), 25, 35),
        Item(Ellipse((0,0), 0.5, 1), 25, 35),
        Item(Polygon([(0,0), (2,0), (2,2), (0,2)]), 25, 35),
        Item(Ellipse((0,0), 1, 1), 25, 35)
    ]
    problem = Problem(container, items)
    solution = Solution(problem)
    problems.append(problem)
    solutions.append(solution)
    positions = [(1,1), (3,1), (5,1), (7,1), (4,4)]
    angles = [0, 0, 90, 0, 45]
    for i, (pos, angle) in enumerate(zip(positions, angles)):
        print_if_allowed(solution.add_item(i, pos, angle), can_print)

    # Vấn đề 6: Hình chữ nhật chứa hỗn hợp hình tròn, ellipse và vuông
    logger.info("Creating problem 6: Rectangle containing mix of circles, ellipses and squares")
    container = Container(250, Polygon([(0,0), (12,0), (12,6), (0,6)]))
    items = [
        Item(Circle((0,0), 1), 40, 50),
        Item(Ellipse((0,0), 1.2, 0.8), 35, 45),
        Item(Polygon([(0,0), (2,0), (2,2), (0,2)]), 45, 55),
        Item(Ellipse((0,0), 0.8, 1.2), 30, 40),
        Item(Circle((0,0), 0.9), 35, 45)
    ]
    problem = Problem(container, items)
    solution = Solution(problem)
    problems.append(problem)
    solutions.append(solution)
    positions = [(1,3), (3,3), (6,3), (9,3), (11,3)]
    angles = [0, 45, 0, 90, 0]
    for i, (pos, angle) in enumerate(zip(positions, angles)):
        print_if_allowed(solution.add_item(i, pos, angle), can_print)

    # Vấn đề 7: Hình đa giác phức tạp chứa các ellipse xoay
    logger.info("Creating problem 7: Complex polygon containing rotated ellipses")
    container = Container(300, Point(5,5).buffer(4, 6))
    items = [Item(Ellipse((0,0), 0.8, 0.4), 50, 60) for _ in range(5)]
    problem = Problem(container, items)
    solution = Solution(problem)
    problems.append(problem)
    solutions.append(solution)
    positions = [(3,3), (3,7), (7,3), (7,7), (5,5)]
    angles = [0, 45, 90, 135, 180]
    for i, (pos, angle) in enumerate(zip(positions, angles)):
        print_if_allowed(solution.add_item(i, pos, angle), can_print)

    # Vấn đề 8: Hình ellipse cao chứa các hình chữ nhật nhỏ
    logger.info("Creating problem 8: Tall ellipse containing small rectangles")
    container = Container(220, Ellipse((2,6), 2, 6))
    items = [Item(Polygon([(0,0), (1.5,0), (1.5,0.8), (0,0.8)]), 35, 45) for _ in range(5)]
    problem = Problem(container, items)
    solution = Solution(problem)
    problems.append(problem)
    solutions.append(solution)
    positions = [(1,2), (1,4), (1,6), (1,8), (1,10)]
    for i, pos in enumerate(positions):
        print_if_allowed(solution.add_item(i, pos, 0), can_print)

    # Vấn đề 9: Hình tròn lớn chứa các ellipse xoay
    logger.info("Creating problem 9: Large circle containing rotated ellipses")
    container = Container(280, Circle((6,6), 6))
    items = [Item(Ellipse((0,0), 1.2, 0.6), 45, 55) for _ in range(5)]
    problem = Problem(container, items)
    solution = Solution(problem)
    problems.append(problem)
    solutions.append(solution)
    positions = [(4,4), (4,8), (8,4), (8,8), (6,6)]
    angles = [0, 45, 90, 135, 180]
    for i, (pos, angle) in enumerate(zip(positions, angles)):
        print_if_allowed(solution.add_item(i, pos, angle), can_print)

    # Vấn đề 10: Hình chữ nhật rộng chứa hỗn hợp các hình
    logger.info("Creating problem 10: Wide rectangle containing mix of shapes")
    container = Container(350, Polygon([(0,0), (15,0), (15,5), (0,5)]))
    items = [
        Item(Ellipse((0,0), 1.2, 0.8), 60, 70),
        Item(Circle((0,0), 1), 55, 65),
        Item(Polygon([(0,0), (2,0), (2,1.5), (0,1.5)]), 50, 60),
        Item(Ellipse((0,0), 0.8, 1.2), 45, 55),
        Item(Circle((0,0), 0.9), 40, 50)
    ]
    problem = Problem(container, items)
    solution = Solution(problem)
    problems.append(problem)
    solutions.append(solution)
    positions = [(1.5,2.5), (4.5,2.5), (7.5,2.5), (10.5,2.5), (13.5,2.5)]
    angles = [0, 0, 0, 90, 0]
    for i, (pos, angle) in enumerate(zip(positions, angles)):
        print_if_allowed(solution.add_item(i, pos, angle), can_print)

    logger.info("Finished creating all problems")
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

    logger.info(f"Algorithm {algorithm.__name__} completed in {elapsed_time}ms with solution value {solution.value}")
    return solution, solution.value, value_evolution, elapsed_time, None


def execute_algorithm(algorithm, problem, execution_num=1, calculate_times=False, calculate_value_evolution=False):
    logger.info(f"Starting {execution_num} executions of algorithm {algorithm.__name__}")
    param_tuples = [(algorithm, problem, calculate_times, calculate_value_evolution) for _ in range(execution_num)]
    solutions = []
    values = []
    value_evolutions = []
    times = []
    time_divisions = []

    for i, params in enumerate(param_tuples):
        logger.info(f"Starting execution {i+1}/{execution_num}")
        solution, value, value_evolution, elapsed_time, time_division = execute_algorithm_with_params(params)
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
    problems, problem_names, manual_solutions = create_knapsack_packing_problems_with_manual_solutions()

    experiment_dict = {}

    for problem, problem_name, solution in zip(problems, problem_names, manual_solutions):
        logger.info(f"Processing problem {problem_name}")
        experiment_dict[problem_name] = {
            "problem": problem,
            "manual_solution": solution,
            "algorithms": {}
        }

        for algorithm_name, algorithm in [("Tham lam", greedy.solve_problem), ("Quy hoạch động", dp.solve_problem), ("Di truyền", evolutionary.solve_problem)]:
            logger.info(f"Running algorithm {algorithm_name} on problem {problem_name}")
            solutions, values, value_evolutions, times, time_divisions = execute_algorithm(
                algorithm=algorithm,
                problem=problem,
                execution_num=10,
                calculate_times=True,
                calculate_value_evolution=True
            )
            experiment_dict[problem_name]["algorithms"][algorithm_name] = {
                "solutions": solutions,
                "values": values,
                "value_evolutions": value_evolutions,
                "times": times,
                "time_divisions": time_divisions}
            logger.info(f"Completed algorithm {algorithm_name} on problem {problem_name}")

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
        save_aggregated_result_tables=True
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
            "Cont. area satur. %"
        ]

        data_frame = pd.DataFrame(index=problem_names, columns=fields)

        for problem_name in experiment_dict.keys():
            problem = experiment_dict[problem_name]["problem"]
            solution = experiment_dict[problem_name]["manual_solution"]
            if type(solution) == Solution:
                problem_results = [
                    len(problem.items),
                    round(len(solution.placed_items) / len(problem.items) * 100, 2),
                    round(sum([problem.items[item_index].value for item_index in solution.placed_items.keys()]) / sum([item.value for item in problem.items]) * 100, 2),
                    round(sum([item.weight for item in problem.items]) / problem.container.max_weight * 100, 2),
                    round(sum([item.shape.area for item in problem.items]) / problem.container.shape.area * 100, 2),
                    round(sum([problem.items[item_index].weight for item_index in solution.placed_items.keys()]) / problem.container.max_weight * 100, 2),
                    round(sum([problem.items[item_index].shape.area for item_index in solution.placed_items.keys()]) / problem.container.shape.area * 100, 2),
                ]
                data_frame.loc[problem_name] = problem_results

        if len(data_frame) > 0:
            data_frame.index = [("Problem " + name if len(name) < 5 else name) for name in data_frame.index]
            min_row = data_frame.min()
            max_row = data_frame.max()
            mean_row = data_frame.mean()
            std_row = data_frame.std()
            data_frame.loc["Min"] = round(min_row, 2)
            data_frame.loc["Max"] = round(max_row, 2)
            data_frame.loc["Mean"] = round(mean_row, 2)
            data_frame.loc["Std"] = round(std_row, 2)
            if (max_row != min_row).all():
                data_frame.loc["Std / (max - min) %"] = round(std_row / (max_row - min_row) * 100, 2)
            else:
                data_frame.loc["Std / (max - min) %"] = float('inf')

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
    if show_manual_solution_plots or save_manual_solution_plots or show_algorithm_solution_plots or save_algorithm_solution_plots or show_algorithm_comparison or save_algorithm_comparison:
        for problem_name in experiment_dict.keys():
            logger.info(f"Processing visualizations for problem {problem_name}")

            problem, manual_solution, algorithm_dict = experiment_dict[problem_name].values()

            # create a subdirectory to store the solutions of the problem (if not done yet)
            problem_dir_path = output_dir + problem_name + "/"
            if not os.path.exists(problem_dir_path):
                os.mkdir(problem_dir_path)
                logger.info(f"Created problem directory: {problem_dir_path}")

            plotted_problem_name = "Problem " + problem_name if len(problem_name) < 5 else problem_name

            # if needed, show/save a plot of the initial state (empty solution), and the final state of a manual solution
            if show_manual_solution_plots or save_manual_solution_plots:
                logger.info(f"Processing manual solution plots for problem {problem_name}")
                empty_solution = Solution(problem)

                empty_solution.visualize(
                    title_override=plotted_problem_name + " - Initial state",
                    show_plot=show_manual_solution_plots,
                    save_path=problem_dir_path + "empty_solution.png" if save_manual_solution_plots else None,
                    show_item_value_and_weight=can_plots_show_value_and_weight,
                    show_value_weight_ratio_bar=can_plots_show_value_and_weight)

                if type(manual_solution) == Solution:
                    manual_solution.visualize(
                        title_override=plotted_problem_name + " - Manual solution", show_plot=show_manual_solution_plots, save_path=problem_dir_path + "manual_solution.png" if save_manual_solution_plots else None, show_item_value_and_weight=can_plots_show_value_and_weight, show_value_weight_ratio_bar=can_plots_show_value_and_weight)

            # if required, show/save plots of the solutions of each algorithm
            if show_algorithm_solution_plots or save_algorithm_solution_plots:
                for algorithm_name, subdict in algorithm_dict.items():
                    logger.info(f"Processing solution plots for algorithm {algorithm_name}")
                    for i, solution in enumerate(subdict["solutions"]):
                        solution.visualize(
                            title_override=plotted_problem_name + " - " + algorithm_name + " solution", show_plot=show_algorithm_solution_plots,
                            save_path=problem_dir_path + "" + algorithm_name.lower() + "_exec" + str(i + 1) + "_solution.png" if save_algorithm_solution_plots else None,
                            show_item_value_and_weight=can_plots_show_value_and_weight,
                            show_value_weight_ratio_bar=can_plots_show_value_and_weight
                        )

            # if required, show/save plots of the value evolution of each algorithm
            if show_value_evolution_plots or save_value_evolution_plots:
                for algorithm_name, subdict in algorithm_dict.items():
                    logger.info(f"Processing value evolution plots for algorithm {algorithm_name}")
                    for i, value_evolution in enumerate(subdict["value_evolutions"]):
                        if value_evolution:
                            if type(value_evolution) == list and type(value_evolution[0]) == list:
                                visualize_boxplot_for_data_sequence(
                                    data_lists=value_evolution,
                                    title=plotted_problem_name + " - Population fitness per generation",
                                    show_plot=show_value_evolution_plots,
                                    save_path=problem_dir_path + "" + algorithm_name.lower() + "_exec" + str(i + 1) + "_fitness_evolution.png" if save_value_evolution_plots else None)
                            else:
                                visualize_plot(
                                    values=value_evolution,
                                    title=plotted_problem_name + " - " + algorithm_name + " solution value per iteration",
                                    show_plot=show_value_evolution_plots,
                                    save_path=problem_dir_path + "" + algorithm_name.lower() + "_exec" + str(i + 1) + "_value_evolution.png" if save_value_evolution_plots else None
                                )

            # if required, show/save plots of the time division in tasks of each algorithm
            if show_time_division_plots or save_time_division_plots:
                for algorithm_name, subdict in algorithm_dict.items():
                    logger.info(f"Processing time division plots for algorithm {algorithm_name}")
                    for i, time_division in enumerate(subdict["time_divisions"]):
                        if time_division and time_division.values():  # Ensure time_division is not None and has values
                            visualize_bar_plot(values=[value_pair[0] for value_pair in time_division.values()], labels=[add_newlines_by_spaces(label, 7) for label in list(time_division.keys())], title="Problem " + problem_name + " - " + algorithm_name +
                                               " time per task (milliseconds)", show_plot=show_algorithm_solution_plots, save_path=problem_dir_path + "" + algorithm_name.lower() + "_exec" + str(i + 1) + "_time_division.png" if save_algorithm_solution_plots else None)

            # if needed, show/save plots that compare the value and time of each algorithm considering multiple executions
            if show_algorithm_comparison or save_algorithm_comparison:
                logger.info(f"Processing algorithm comparison plots for problem {problem_name}")
                visualize_boxplot_for_data_sequence(
                    data_lists=[experiment_dict[problem_name]["algorithms"][algo_name]["values"] for algo_name in experiment_dict[problem_name]["algorithms"].keys(
                    )], title="Problem " + problem_name + " - Solution value by algorithm", labels=experiment_dict[problem_name]["algorithms"].keys(), show_plot=show_algorithm_comparison, save_path=problem_dir_path + "value_comparison.png" if save_algorithm_comparison else None)
                visualize_boxplot_for_data_sequence(data_lists=[experiment_dict[problem_name]["algorithms"][algo_name]["times"] for algo_name in experiment_dict[problem_name]["algorithms"].keys()], title="Problem " + problem_name + " - Computational time (milliseconds) by algorithm",
                                                    labels=experiment_dict[problem_name]["algorithms"].keys(), y_scale_override="log", show_plot=show_algorithm_comparison, save_path=problem_dir_path + "time_comparison.png" if save_algorithm_comparison else None)

    # if needed, save tables with an aggregation of the value and time results of the executions of each problem (or just show them)
    if show_aggregated_result_tables or save_aggregated_result_tables:
        logger.info("Processing aggregated result tables")
        problem_names = list(experiment_dict.keys())
        algorithm_names = [algo_name for algo_name in experiment_dict[problem_names[0]]["algorithms"].keys()]
        fields = ["mean", "std", "min", "med", "max"]
        for concept in ["value", "time"]:
            algo_field_tuples = [(algo_name, field) for algo_name in algorithm_names for field in fields]
            if concept == "value":
                algo_field_tuples += [("Manual", "optim.")]
            multi_index = pd.MultiIndex.from_tuples(algo_field_tuples, names=["Algorithm", "Statistic"])
            data_frame = pd.DataFrame(index=problem_names, columns=multi_index)
            for problem_name in experiment_dict.keys():
                problem_results = list()
                for algo_name in algorithm_names:
                    mean, std, min_, median, max_ = get_stats(experiment_dict[problem_name]["algorithms"][algo_name][concept + "s"], 2 if concept == "value" else 0)
                    problem_results.extend([mean, std, min_, median, max_])
                if concept == "value":
                    if type(experiment_dict[problem_name]["manual_solution"]) == Solution:
                        problem_results.append(experiment_dict[problem_name]["manual_solution"].value)
                    else:
                        problem_results.append(experiment_dict[problem_name]["manual_solution"])
                data_frame.loc[problem_name] = problem_results
            data_frame.index = [("Problem " + name if len(name) < 5 else name) for name in data_frame.index]
            if show_aggregated_result_tables:
                print("{} results:\n{}\n".format(concept.capitalize(), data_frame.to_string()))
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
