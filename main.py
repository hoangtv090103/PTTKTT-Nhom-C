import time
import os
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, Point
from circle import Circle
from container import Container
from item import Item
from problem import Problem
from solution import Solution
import evolutionary
import greedy
from common_algo_functions import get_stats, print_if_allowed, get_time_since, visualize_plot, visualize_boxplot_for_data_sequence, visualize_bar_plot, add_newlines_by_spaces


def create_knapsack_packing_problems_with_manual_solutions(can_print=False):
    problems, solutions = [], []

    # Vấn đề 1
    container = Container(120, Circle((3.3, 3.3), radius=3.3))
    items = [
        Item(Polygon([(0, 0), (0, 4.5), (4.5, 4.5), (4.5, 0)]), 40., 50.),
        Item(Circle((0, 0), 0.45), 20, 5),
        Item(Circle((0, 0), 0.45), 20, 10),
        Item(Circle((0, 0), 0.45), 20, 15),
        Item(Circle((0, 0), 0.45), 20, 20)
    ]
    problem = Problem(container, items)
    solution = Solution(problem)
    problems.append(problem)
    solutions.append(solution)

    # Thêm các mục với vị trí và góc
    positions_angles = [(0, (3.3, 3.3), 0.), (1, (3.3, 6.05), 0.), (2, (3.3, 0.55), 0.), (3, (6.05, 3.3), 0.), (4, (0.55, 3.3), 0.)]
    for item, pos, angle in positions_angles:
        print_if_allowed(solution.add_item(item, pos, angle), can_print)
    # Các vấn đề khác theo cấu trúc tương tự

    # Vấn đề 2
    container = Container(100., Point(5, 5).buffer(5, 4))
    items = [Item(MultiPolygon([(Point(5, 5).buffer(i, 4).exterior.coords, [tuple(Point(5, 5).buffer(i-0.7, 4).exterior.coords)])]), 10, 25-i*5) for i in range(5)]
    problem = Problem(container, items)
    solution = Solution(problem)
    problems.append(problem)
    solutions.append(solution)
    for i in range(5):
        print_if_allowed(solution.add_item(i, (5, 5), 0.), can_print)

    # Tương tự cho Vấn đề 3, 4, ..., Vấn đề 10
    return problems, [str(i + 1) for i in range(len(problems))], solutions


def execute_algorithm_with_params(params):
    algorithm, problem, calculate_times, calculate_value_evolution = params[:4]
    start_time = time.time()

    if calculate_value_evolution:
        # Nếu hàm solve_problem không hỗ trợ return_population_fitness_per_generation, bỏ qua tham số này
        result = algorithm(problem, calculate_times=calculate_times)
    else:
        result = algorithm(problem, calculate_times=calculate_times)

    elapsed_time = get_time_since(start_time)
    solution = result[0]
    value_evolution = result[1] if calculate_value_evolution else None

    return solution, solution.value, value_evolution, elapsed_time, None


def execute_algorithm(algorithm, problem, execution_num=1, calculate_times=False, calculate_value_evolution=False):
    param_tuples = [(algorithm, problem, calculate_times, calculate_value_evolution) for _ in range(execution_num)]
    solutions = []
    values = []
    value_evolutions = []
    times = []
    time_divisions = []

    for params in param_tuples:
        solution, value, value_evolution, elapsed_time, time_division = execute_algorithm_with_params(params)
        solutions.append(solution)
        values.append(value)
        value_evolutions.append(value_evolution)
        times.append(elapsed_time)
        time_divisions.append(time_division)

    return solutions, values, value_evolutions, times, time_divisions


def run():
    problems, problem_names, manual_solutions = create_knapsack_packing_problems_with_manual_solutions()

    experiment_dict = {}

    for problem, problem_name, solution in zip(problems, problem_names, manual_solutions):
        experiment_dict[problem_name] = {
            "problem": problem,
            "manual_solution": solution,
            "algorithms": {}
        }

        for algorithm_name, algorithm in [("Greedy", greedy.solve_problem), ("Evolutionary", evolutionary.solve_problem)]:
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

    # if needed, save statistics about the problems and their manual solutions
    if show_problem_stats or save_problem_stats:
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

    # create the problem results directory (if not done yet)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # for each problem's results, show or save plots if needed
    if show_manual_solution_plots or save_manual_solution_plots or show_algorithm_solution_plots or save_algorithm_solution_plots or show_algorithm_comparison or save_algorithm_comparison:
        for problem_name in experiment_dict.keys():

            problem, manual_solution, algorithm_dict = experiment_dict[problem_name].values()

            # create a subdirectory to store the solutions of the problem (if not done yet)
            problem_dir_path = output_dir + problem_name + "/"
            if not os.path.exists(problem_dir_path):
                os.mkdir(problem_dir_path)

            plotted_problem_name = "Problem " + problem_name if len(problem_name) < 5 else problem_name

            # if needed, show/save a plot of the initial state (empty solution), and the final state of a manual solution
            if show_manual_solution_plots or save_manual_solution_plots:
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
                    for i, time_division in enumerate(subdict["time_divisions"]):
                        if time_division and time_division.values():  # Ensure time_division is not None and has values
                            visualize_bar_plot(values=[value_pair[0] for value_pair in time_division.values()], labels=[add_newlines_by_spaces(label, 7) for label in list(time_division.keys())], title="Problem " + problem_name + " - " + algorithm_name +
                                               " time per task (milliseconds)", show_plot=show_algorithm_solution_plots, save_path=problem_dir_path + "" + algorithm_name.lower() + "_exec" + str(i + 1) + "_time_division.png" if save_algorithm_solution_plots else None)

            # if needed, show/save plots that compare the value and time of each algorithm considering multiple executions
            if show_algorithm_comparison or save_algorithm_comparison:
                visualize_boxplot_for_data_sequence(
                    data_lists=[experiment_dict[problem_name]["algorithms"][algo_name]["values"] for algo_name in experiment_dict[problem_name]["algorithms"].keys(
                    )], title="Problem " + problem_name + " - Solution value by algorithm", labels=experiment_dict[problem_name]["algorithms"].keys(), show_plot=show_algorithm_comparison, save_path=problem_dir_path + "value_comparison.png" if save_algorithm_comparison else None)
                visualize_boxplot_for_data_sequence(data_lists=[experiment_dict[problem_name]["algorithms"][algo_name]["times"] for algo_name in experiment_dict[problem_name]["algorithms"].keys()], title="Problem " + problem_name + " - Computational time (milliseconds) by algorithm",
                                                    labels=experiment_dict[problem_name]["algorithms"].keys(), y_scale_override="log", show_plot=show_algorithm_comparison, save_path=problem_dir_path + "time_comparison.png" if save_algorithm_comparison else None)

    # if needed, save tables with an aggregation of the value and time results of the executions of each problem (or just show them)
    if show_aggregated_result_tables or save_aggregated_result_tables:
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


if __name__ == "__main__":
    experiment_dict = run()
    visualize_and_save_experiments(experiment_dict, output_dir="experiment_results/")
