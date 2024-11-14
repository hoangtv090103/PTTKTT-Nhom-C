import copy
import itertools
import math
import random
import time
import numpy as np
from common_algo_functions import *
import greedy
from circle import Circle
from utils import *
from ellipse import Ellipse

# kích thước của quần thể
POPULATION_SIZE = 100

# kích thước của quần thể con được tạo ra trong mỗi thế hệ
OFFSPRING_SIZE = POPULATION_SIZE * 2

# kích thước của tập hợp quần thể ưu tú của thuật toán
ELITE_SIZE = 5

# kích thước của nhóm đấu loại được sử dụng để chọn cha mẹ
PARENT_SELECTION_POOL_SIZE = 3

# kích thước của nhóm đấu loại được sử dụng để cập nhật quần thể
POPULATION_UPDATE_POOL_SIZE = 3

# số vòng lặp tối đa để tạo ra một giải pháp ban đầu
INITIAL_SOLUTION_GENERATION_MAX_ITER_NUM = 300

# số vòng lặp tối đa để giả định rằng một giải pháp ban đầu đã hội tụ
INITIAL_SOLUTION_GENERATION_CONVERGE_ITER_NUM = 50

# số vòng lặp tối thiểu của đột biến
MUTATION_MIN_ITER_NUM = 5

# trọng số xác suất để thêm một mục trong bước đột biến
MUTATION_ADD_WEIGHT = 0.6

# trọng số xác suất để loại bỏ một mục trong bước đột biến
MUTATION_REMOVE_WEIGHT = 0.1

# trọng số xác suất để thay đổi vị trí của một mục trong bước đột biến
MUTATION_MODIFY_WEIGHT = 0.3

# số lần tối đa để thử thêm một mục cụ thể trong bước đột biến
MUTATION_ADD_MAX_ATTEMPT_NUM = 10

# số lần tối đa để thử thay đổi vị trí của một mục cụ thể trong bước đột biến
MUTATION_MODIFY_MAX_ATTEMPT_NUM = 3

# tỷ lệ tối đa của chiều dài cạnh của hộp chứa (cho mỗi trục) có thể được áp dụng như là một giá trị dịch chuyển trong bước đột biến của việc thay đổi vị trí nhỏ
MUTATION_MODIFY_SMALL_POSITION_CHANGE_PROPORTION = 0.2

# tỷ lệ tối đa của 360 độ có thể được áp dụng như là một giá trị dịch chuyển quay trong bước đột biến của việc thay đổi góc quay nhỏ
MUTATION_MODIFY_SMALL_ROTATION_CHANGE_PROPORTION = 0.2

# số vị trí tối đa cần kiểm tra trong bước đột biến thay đổi vị trí nhằm di chuyển một mục theo hướng cho đến khi giao nhau
MUTATION_MODIFY_MOVE_UNTIL_INTERSECTION_POINT_NUM = 15

# tỷ lệ chiều dài cạnh của hình chữ nhật bao quanh của hộp chứa được sử dụng để tính khoảng cách tối thiểu có liên quan giữa điểm tham chiếu của một mục và điểm giao nhau đầu tiên xác định bởi hướng di chuyển-trong-cho đến khi giao nhau
MUTATION_MODIFY_MOVE_UNTIL_INTERSECTION_MIN_DIST_PROPORTION = 0.03

# số góc tối đa cần kiểm tra trong bước đột biến thay đổi góc nhằm xoay một mục theo hướng cho đến khi giao nhau
MUTATION_MODIFY_ROTATE_UNTIL_INTERSECTION_ANGLE_NUM = 8

# xác suất chọn giải pháp trung gian làm kết quả của đột biến nếu nó tốt hơn giải pháp kết quả từ bước cuối cùng
MUTATION_INTERMEDIATE_SELECTION_PROB = 1

# có sử dụng phép lai tạo hay không
CAN_USE_CROSSOVER = False

# xác suất bỏ qua đột biến của một giải pháp được tạo ra từ phép lai tạo nếu nó tốt hơn (có độ phù hợp cao hơn) trước khi đột biến
CROSSOVER_IGNORE_MUTATION_PROBABILITY = 0.5

# số lần thử tối đa để tạo ra một hình dạng phân chia cho phép lai tạo mà giao cắt với hộp chứa (phủ một phần diện tích của nó)
CROSSOVER_MAX_ATTEMPT_NUM = 5

# tỷ lệ tối thiểu của chiều dài cạnh của hình chữ nhật bao quanh của hộp chứa (cho trục liên quan) có thể được sử dụng để xác định chiều dài (trong cùng trục đó) của một hình phân đoạn không cắt trong phép lai tạo
CROSSOVER_SHAPE_MIN_LENGTH_PROPORTION = 0.25

# tỷ lệ tối đa của chiều dài cạnh của hình chữ nhật bao quanh của hộp chứa (cho trục liên quan) có thể được sử dụng để xác định chiều dài (trong cùng trục đó) của một hình phân đoạn không cắt trong phép lai tạo
CROSSOVER_SHAPE_MAX_LENGTH_PROPORTION = 0.75

# số đỉnh tối đa có thể được sử dụng để tạo ra một đa giác để phân chia không gian hộp chứa của cha mẹ thành hai khu vực khi thực hiện phép lai tạo
CROSSOVER_POLYGON_MAX_VERTEX_NUM = 10

# số hoán vị tối đa xác định thứ tự khả dĩ mà các mục nằm một phần trong cả hai khu vực của phép phân chia hộp chứa có thể được thử đặt lại vào hộp chứa
CROSSOVER_MAX_PERMUTATION_NUM = 5

# tỷ lệ cá thể trong một quần thể mà các giải pháp ứng viên được tạo ra từ các hoán vị của phép lai tạo (cho các mục nằm trong cả hai khu vực của phép phân chia hộp chứa) cần phải vượt trội về độ phù hợp để đủ điều kiện trong lần cập nhật quần thể tiếp theo
CROSSOVER_MIN_FITNESS_FOR_NON_BEST_PROPORTION = 0.95

# tỷ lệ tối thiểu của diện tích hộp chứa mà một hình dạng có vùng kín cần phải có để được chấp nhận như là một hình phân chia hợp lệ trong phép lai tạo
CROSSOVER_SHAPE_MIN_AREA_PROPORTION = 0.1

# số thế hệ tối đa
MAX_GENERATION_NUM = 30

# số thế hệ tối đa để giả định rằng thuật toán đã hội tụ
CONVERGE_GENERATION_NUM = 12

# tỷ lệ tối đa của số vòng lặp trong việc tạo ra giải pháp ban đầu trong đó chuyên môn hóa trong việc cố gắng đặt một mục cụ thể
INITIAL_SOLUTION_GENERATION_FIRST_ITEM_SPECIALIZATION_ITER_PROPORTION = 0.5


def get_fitness(solution):
    """Trả về giá trị độ phù hợp của giải pháp đã cho"""

    if not solution:
        return 0

    # độ phù hợp của một giải pháp hợp lệ trùng với giá trị của nó
    return solution.value


def get_fittest_solution(solutions):
    """Trả về giải pháp có độ phù hợp cao nhất trong số các giải pháp đã cho, sử dụng tiêu chí phá vỡ hòa nếu có hòa và chỉ sử dụng phá vỡ hòa ngẫu nhiên như là phương án cuối cùng"""

    # đảm bảo rằng có các giải pháp
    if not solutions:
        return None

    # một giải pháp duy nhất là giải pháp có độ phù hợp cao nhất
    if len(solutions) == 1:
        return solutions[0]

    fittest_solutions = list()
    max_fitness = -np.inf

    # tìm các giải pháp có độ phù hợp cao nhất
    for solution in solutions:
        fitness = get_fitness(solution)

        if fitness > max_fitness:
            fittest_solutions = [solution]
            max_fitness = fitness
        elif fitness == max_fitness:
            fittest_solutions.append(solution)

    # có thể không có giải pháp có độ phù hợp cao nhất
    if not fittest_solutions:
        return None

    # nếu chỉ có một giải pháp có độ phù hợp cao nhất, trả về giải pháp đó
    if len(fittest_solutions) == 1:
        return fittest_solutions[0]

    # đối với các trường hợp hòa, sử dụng tối ưu hóa diện tích nhỏ nhất làm tiêu chí phá vỡ hòa
    min_area_solutions = list()
    min_area = np.inf
    for solution in fittest_solutions:
        area = solution.get_area()

        if area < min_area:
            min_area_solutions = [solution]
            min_area = area
        elif area == min_area:
            min_area_solutions.append(solution)

    # có thể không có giải pháp phá vỡ hòa; trả về một giải pháp ngẫu nhiên trong số các giải pháp có độ phù hợp cao nhất
    if not min_area_solutions:
        return random.choice(fittest_solutions)

    # nếu chỉ có một giải pháp phá vỡ hòa, trả về giải pháp đó
    if len(min_area_solutions) == 1:
        return min_area_solutions[0]

    # nếu hòa vẫn tiếp tục, sử dụng tối ưu hóa diện tích hình chữ nhật bao quanh toàn cầu nhỏ nhất làm tiêu chí phá vỡ hòa
    min_bound_area_solutions = list()
    min_bound_area = np.inf
    for solution in min_area_solutions:
        bound_area = solution.get_global_bounding_rectangle_area()

        if bound_area < min_bound_area:
            min_bound_area_solutions = [solution]
            min_bound_area = bound_area
        elif bound_area == min_bound_area:
            min_bound_area_solutions.append(solution)

    # có thể không có giải pháp phá vỡ hòa; trả về một giải pháp ngẫu nhiên trong số các giải pháp có diện tích nhỏ nhất
    if not min_bound_area_solutions:
        return random.choice(min_area_solutions)

    # nếu chỉ có một giải pháp phá vỡ hòa, trả về giải pháp đó
    if len(min_bound_area_solutions) == 1:
        return min_bound_area_solutions[0]

    # nếu hòa vẫn tiếp tục, sử dụng ngẫu nhiên trong số các giải pháp cuối cùng bị hòa
    return random.choice(min_bound_area_solutions)


def get_fittest_solution_with_circle(solutions):
    """Trả về giải pháp có độ phù hợp cao nhất trong số các giải pháp đã cho, sử dụng tiêu chí phá vỡ hòa nếu có hòa và chỉ sử dụng phá vỡ hòa ngẫu nhiên như là phương án cuối cùng"""

    # đảm bảo rằng có các giải pháp
    if not solutions:
        return None

    # một giải pháp duy nhất là giải pháp có độ phù hợp cao nhất
    if len(solutions) == 1:
        return solutions[0]

    fittest_solutions = list()
    max_fitness = -np.inf

    # tìm các giải pháp có độ phù hợp cao nhất
    for solution in solutions:
        fitness = get_fitness(solution)

        if fitness > max_fitness:
            fittest_solutions = [solution]
            max_fitness = fitness
        elif fitness == max_fitness:
            fittest_solutions.append(solution)

    # có thể không có giải pháp có độ phù hợp cao nhất
    if not fittest_solutions:
        return None

    # nếu chỉ có một giải pháp có độ phù hợp cao nhất, trả về giải pháp đó
    if len(fittest_solutions) == 1:
        return fittest_solutions[0]

    # đối với các trường hợp hòa, sử dụng tối ưu hóa diện tích nhỏ nhất làm tiêu chí phá vỡ hòa
    min_area_solutions = list()
    min_area = np.inf
    for solution in fittest_solutions:
        area = solution.get_area()

        if area < min_area:
            min_area_solutions = [solution]
            min_area = area
        elif area == min_area:
            min_area_solutions.append(solution)

    # có thể không có giải pháp phá vỡ hòa; trả về một giải pháp ngẫu nhiên trong số các giải pháp có độ phù hợp cao nhất
    if not min_area_solutions:
        return random.choice(fittest_solutions)

    # nếu chỉ có một giải pháp phá vỡ hòa, trả về giải pháp đó
    if len(min_area_solutions) == 1:
        return min_area_solutions[0]

    # nếu hòa vẫn tiếp tục, sử dụng tối ưu hóa diện tích hình tròn bao quanh nhỏ nhất làm tiêu chí phá vỡ hòa
    min_bound_area_solutions = list()
    min_bound_area = np.inf
    for solution in min_area_solutions:
        bound_area = solution.get_bounding_circle_area()

        if bound_area < min_bound_area:
            min_bound_area_solutions = [solution]
            min_bound_area = bound_area
        elif bound_area == min_bound_area:
            min_bound_area_solutions.append(solution)

    # có thể không có giải pháp phá vỡ hòa; trả về một giải pháp ngẫu nhiên trong số các giải pháp có diện tích nhỏ nhất
    if not min_bound_area_solutions:
        return random.choice(min_area_solutions)

    # nếu chỉ có một giải pháp phá vỡ hòa, trả về giải pháp đó
    if len(min_bound_area_solutions) == 1:
        return min_bound_area_solutions[0]

    # nếu hòa vẫn tiếp tục, sử dụng ngẫu nhiên trong số các giải pháp cuối cùng bị hòa
    return random.choice(min_bound_area_solutions)


def get_fittest_solutions(solutions, size, break_ties=True):
    """Trả về các giải pháp có độ phù hợp cao nhất, số lượng như đã chỉ định"""

    # nếu có ít giải pháp hơn số lượng mong muốn, trả về tất cả chúng
    if len(solutions) <= size:
        return solutions

    # sắp xếp các giải pháp theo độ phù hợp giảm dần
    solutions_by_fitness = sorted(
        solutions, key=lambda solution: get_fitness(solution), reverse=True)

    # chỉ cắt ngắn (nếu cần) danh sách các cá thể đã sắp xếp theo độ phù hợp nếu không cần phá vỡ hòa
    if not break_ties:
        return solutions_by_fitness[:size]

    # nếu mục cuối cùng trong phạm vi kích thước có độ phù hợp khác với mục tiếp theo (mục đầu tiên bị bỏ qua), an toàn để cắt ngắn
    if len(solutions_by_fitness) == size or get_fitness(solutions_by_fitness[size - 1]) != get_fitness(solutions_by_fitness[size]):
        return solutions_by_fitness[:size]

    # nếu không, tìm chỉ số đầu tiên và cuối cùng với độ phù hợp hòa ở cuối
    tied_at_end_fitness = get_fitness(solutions_by_fitness[size])
    start_index = size - 1
    end_index = size
    for i in range(end_index, len(solutions_by_fitness)):
        if i != end_index:
            if get_fitness(solutions_by_fitness[i]) == tied_at_end_fitness:
                end_index = i
            else:
                break
    for i in range(start_index, 0, -1):
        if i != start_index:
            if get_fitness(solutions_by_fitness[i]) == tied_at_end_fitness:
                start_index = i
            else:
                break
    tied_at_end_solutions = solutions_by_fitness[start_index:end_index]
    solutions_by_fitness = solutions_by_fitness[:start_index]

    while len(solutions_by_fitness) != size:
        # tie_breaker_solution = get_fittest_solution(tied_at_end_solutions)
        tie_breaker_solution = get_fittest_solution_with_circle(
            tied_at_end_solutions)
        tied_at_end_solutions.remove(tie_breaker_solution)
        solutions_by_fitness.append(tie_breaker_solution)

    return solutions_by_fitness


def generate_initial_solution(problem, item_index_to_place_first=-1, item_specialization_iter_proportion=0., calculate_times=False):
    """Tạo ra một giải pháp ban đầu cho vấn đề đã cho bằng cách cố gắng đặt các mục được chọn ngẫu nhiên cho đến khi một tiêu chí kết thúc được đáp ứng"""

    # sử dụng thuật toán tham lam không có trọng số, với các lựa chọn hoàn toàn ngẫu nhiên
    return greedy.solve_problem(problem, greedy_score_function=greedy.get_constant_score, repetition_num=1, max_iter_num=INITIAL_SOLUTION_GENERATION_MAX_ITER_NUM, max_iter_num_without_changes=INITIAL_SOLUTION_GENERATION_CONVERGE_ITER_NUM, item_index_to_place_first=item_index_to_place_first, item_specialization_iter_proportion=item_specialization_iter_proportion, calculate_times=calculate_times)


def generate_population(problem, population_size, item_specialization_iter_proportion):
    """Tạo ra một quần thể có kích thước đã cho cho vấn đề đã cho"""

    # tìm các mục có trọng lượng không vượt quá sức chứa của hộp chứa
    feasible_item_indices = [index for index, item in enumerate(
        problem.items) if item.weight <= problem.container.max_weight]

    # giới hạn số lượng mục ban đầu cho kích thước quần thể (để mỗi mục được giữ lại có thể có ít nhất một giải pháp chuyên môn hóa), chọn một mẫu ngẫu nhiên của chúng nếu vượt quá
    if len(feasible_item_indices) > population_size:
        random.shuffle(feasible_item_indices)
        feasible_item_indices = feasible_item_indices[:population_size]

    # để thúc đẩy sự đa dạng hóa các giải pháp trong không gian tìm kiếm, mỗi mục khả thi sẽ có cùng số lượng giải pháp mà nó được cố gắng đặt trước tiên, trước bất kỳ mục nào khác, để thúc đẩy rằng tất cả các mục xuất hiện ít nhất trong một số giải pháp ban đầu
    solution_num_per_item_specialization = population_size // len(
        feasible_item_indices)

    population = list()

    # đối với mỗi mục khả thi, khởi tạo một số lượng giải pháp nhất định với mục đó được đặt trước tiên (nếu có thể)
    for item_index in feasible_item_indices:

        population.extend([generate_initial_solution(problem, item_index, item_specialization_iter_proportion)
                          for _ in range(solution_num_per_item_specialization)])

    # tạo ra nhiều giải pháp với khởi tạo tiêu chuẩn như cần thiết để đạt được kích thước quần thể mong muốn
    remaining_solution_num = population_size - len(population)
    population.extend([generate_initial_solution(problem)
                      for _ in range(remaining_solution_num)])

    return population


def get_tournament_winner(population, pool_size, individual_to_ignore=None):
    """Chọn và trả về cá thể chiến thắng (về độ phù hợp) trong số một nhóm cá thể được chọn ngẫu nhiên từ quần thể"""

    # bỏ qua một cá thể nếu cần
    if individual_to_ignore:
        population = population[:]
        population.remove(individual_to_ignore)

    # kích thước nhóm thực tế có thể bị thu nhỏ nếu quần thể quá nhỏ
    pool_size = min(pool_size, len(population))

    # nếu kích thước nhóm là 0, không thể tìm thấy người chiến thắng
    if pool_size == 0:
        return None

    # nếu không, tìm các cá thể của nhóm, cho phép lặp lại
    pool = [random.choice(population) for _ in range(pool_size)]

    # cá thể có độ phù hợp cao nhất là người chiến thắng
    # return get_fittest_solution(pool)
    return get_fittest_solution_with_circle(pool)


def select_parents(population, offspring_size, pool_size, can_use_crossover):
    """Chọn (với một nhóm đấu loại có kích thước đã cho) và trả về các cha mẹ trong số quần thể đã cho để họ có thể tạo ra số lượng con đã cho"""

    # nếu phép lai tạo sẽ được sử dụng, các cặp cha mẹ sẽ được chọn
    if can_use_crossover:
        selection_num = offspring_size // 2
        if offspring_size % 2 != 0:
            selection_num += 1

    # đối với các trường hợp không lai tạo, chỉ một con sẽ được tạo ra cho mỗi cha mẹ, sử dụng đột biến
    else:
        selection_num = offspring_size

    parents = list()

    # chọn các cha mẹ
    for i in range(selection_num):

        # đối với phép lai tạo, các cha mẹ được biểu diễn dưới dạng cặp, và họ không thể là cùng một cá thể
        if can_use_crossover:
            parent0 = get_tournament_winner(population, pool_size)
            if parent0:
                parent1 = get_tournament_winner(population, pool_size, parent0)
                if parent1:
                    parents.append((parent0, parent1))

        # khi không có phép lai tạo, chỉ có đột biến, các cha mẹ được chọn và biểu diễn riêng lẻ
        else:
            parent = get_tournament_winner(population, pool_size)
            if parent:
                parents.append(parent)

    return parents


def get_crossover(parent0, parent1, max_attempt_num, shape_min_length_proportion, shape_max_length_proportion, shape_min_area_proportion, polygon_max_vertex_num, max_permutation_num, min_fitness_for_non_best):
    """Tạo và trả về hai cá thể con từ các cha mẹ đã cho"""

    # xác định các giới hạn và chiều dài cạnh của hộp chứa (chung cho cả hai cha mẹ)
    min_x, min_y, max_x, max_y = get_bounds(parent0.problem.container.shape)
    x_length = abs(min_x - max_x)
    y_length = abs(min_y - max_y)

    # xác định chiều dài tối thiểu và tối đa trong mỗi chiều
    min_x_length = shape_min_length_proportion * x_length
    max_x_length = shape_max_length_proportion * x_length
    min_y_length = shape_min_length_proportion * y_length
    max_y_length = shape_max_length_proportion * y_length

    # xác định diện tích tối thiểu của hình dạng dựa trên diện tích của hộp chứa
    shape_min_area = shape_min_area_proportion * \
        parent0.problem.container.shape.area

    partition_indices = [segment_index, _] = [0, 1]

    shape = None
    has_valid_shape = False
    i = 0

    # thực hiện tối đa một số lần thử để tạo ra một hình dạng phân chia mà ít nhất là một phần nằm trong hộp chứa
    while i < max_attempt_num and not has_valid_shape:

        # chọn ngẫu nhiên tùy chọn phân chia
        partition_index = random.choice(partition_indices)

        # tạo ra một tam giác/tứ giác nếu lần phân chia là lần đầu tiên
        if partition_index == segment_index:
            if bool(random.getrandbits(1)):
                shape = create_random_triangle_in_rectangle_corner(
                    min_x, min_y, max_x, max_y)  # tạo ra một tam giác
            else:
                shape = create_random_quadrilateral_in_rectangle_corners(
                    min_x, min_y, max_x, max_y)

        # tạo ra một hình dạng khác
        else:

            # xác định ngẫu nhiên chiều dài thực tế trong mỗi chiều để sử dụng cho một hình dạng
            shape_x_length = random.uniform(min_x_length, max_x_length)
            shape_y_length = random.uniform(min_y_length, max_y_length)

            shape_indices = [circle_index, ellipse_index, _] = [0, 1, 2]

            # chọn một loại hình dạng
            shape_index = random.choice(shape_indices)

            # tạo ra một hình tròn
            if shape_index == circle_index:
                center = (random.uniform(min_x, max_x),
                          random.uniform(min_y, max_y))
                radius = random.choice([shape_x_length, shape_y_length]) * 0.5
                shape = Circle(center, radius)

            # tạo ra một hình elip
            elif shape_index == ellipse_index:
                center = (random.uniform(min_x, max_x),
                          random.uniform(min_y, max_y))
                shape = Ellipse(center, shape_x_length, shape_y_length)

            # tạo ra một đa giác
            else:
                # chọn ngẫu nhiên số đỉnh cho đa giác (ít nhất là 3, để có một tam giác)
                vertex_num = random.randint(3, polygon_max_vertex_num)

                # chọn ngẫu nhiên một trung tâm cho hình chữ nhật bao quanh của đa giác
                polygon_bounds_center = (random.uniform(
                    min_x, max_x), random.uniform(min_y, max_y))

                # xác định các giới hạn của đa giác
                polygon_min_x = polygon_bounds_center[0] - shape_x_length * 0.5
                polygon_max_x = polygon_min_x + shape_x_length
                polygon_min_y = polygon_bounds_center[1] - shape_y_length * 0.5
                polygon_max_y = polygon_min_y + shape_y_length

                # tạo ra đa giác
                shape = create_random_polygon(
                    polygon_min_x, polygon_min_y, polygon_max_x, polygon_max_y, vertex_num)

        # xác định xem hình dạng hiện tại có hợp lệ hay không (nó phải có diện tích tối thiểu và giao với hộp chứa)
        has_valid_shape = shape is not None and shape.area >= shape_min_area and do_shapes_intersect(
            parent0.problem.container.shape, shape)

        i += 1

    # ban đầu, mỗi cá thể con được tạo ra như là bản sao của một cha mẹ, trước khi tái tổ hợp
    offspring0 = copy.deepcopy(parent0)
    offspring1 = copy.deepcopy(parent1)

    # tái tổ hợp là có thể nếu có một hình dạng phân chia hợp lệ
    if has_valid_shape:

        # sử dụng hình dạng đã tạo để chia các mục đã đặt của mỗi cha mẹ thành 3 danh sách: các mục trong vùng đầu tiên (bên trong hình dạng), các mục trong vùng thứ hai (bên ngoài hình dạng), và các mục giao nhau
        parent0_separated_item_indices = list()
        parent1_separated_item_indices = list()
        for parent in [parent0, parent1]:
            region0_indices, region1_indices, intersected_indices = list(), list(), list()
            for item_index, placed_item in parent.placed_items.items():
                has_intersection = do_shapes_intersect(
                    shape, placed_item.shape)
                if has_intersection:
                    intersected_indices.append(item_index)
                else:
                    if does_shape_contain_other(shape, placed_item.shape):
                        region0_indices.append(item_index)
                    else:
                        region1_indices.append(item_index)
            if parent == parent0:
                parent0_separated_item_indices = [
                    region0_indices, region1_indices, intersected_indices]
            else:
                parent1_separated_item_indices = [
                    region0_indices, region1_indices, intersected_indices]

        # đối với cá thể con đầu tiên, chỉ giữ lại các mục được đặt trong vùng đầu tiên của cha mẹ đầu tiên, sau đó cố gắng đặt các mục của vùng thứ hai của cha mẹ thứ hai (sự trùng lặp được ngăn chặn nội bộ)
        for item_index in parent0_separated_item_indices[1] + parent0_separated_item_indices[2]:
            offspring0.remove_item(item_index)
        for item_index in parent1_separated_item_indices[1]:
            offspring0.add_item(item_index, (random.uniform(
                min_x, max_x), random.uniform(min_y, max_y)), random.uniform(0, 360))

        # đối với cá thể con thứ hai, chỉ giữ lại các mục được đặt trong vùng thứ hai của cha mẹ đầu tiên, sau đó cố gắng đặt các mục của vùng đầu tiên của cha mẹ thứ hai (sự trùng lặp được ngăn chặn nội bộ)
        for item_index in parent0_separated_item_indices[0] + parent0_separated_item_indices[2]:
            offspring1.remove_item(item_index)
        for item_index in parent1_separated_item_indices[0]:
            offspring1.add_item(item_index, (random.uniform(
                min_x, max_x), random.uniform(min_y, max_y)), random.uniform(0, 360))

        # tìm tất cả các cặp (chỉ mục mục, chỉ mục cha mẹ); một cho mỗi mục giao nhau trong một cha mẹ
        item_parent_pairs = list()
        for parent_index, intersected_item_indices in enumerate([parent0_separated_item_indices[2], parent1_separated_item_indices[2]]):
            for item_index in intersected_item_indices:
                item_parent_pairs.append((item_index, parent_index))

        # các hành động tiếp theo chỉ cần thiết nếu có các mục giao nhau
        if item_parent_pairs:

            # tạo ra các hoán vị của tất cả (hoặc một số lượng lớn hợp lý) các thứ tự có thể của các lần thử đặt các mục đã bị cắt (không xử lý sự trùng lặp, vì hàm thêm sẽ loại bỏ chúng ngay lập tức)
            permutations = list(itertools.islice(itertools.permutations(
                item_parent_pairs), max_permutation_num * 10))

            # nếu có nhiều hoán vị hơn số lượng tối đa có thể, lấy một tập hợp con ngẫu nhiên
            if len(permutations) > max_permutation_num:
                permutations = random.choices(
                    permutations, k=max_permutation_num)

            # mỗi hoán vị tạo ra một cá thể con khác nhau (dựa trên một trong hai cá thể con cơ bản, được chọn ngẫu nhiên) cố gắng thêm các mục theo thứ tự của hoán vị; tìm cá thể tốt nhất (cho mỗi cá thể con cơ bản) và tất cả những cá thể có giá trị độ phù hợp đáng kể
            candidates = list()
            high_fitness_candidate_indices = list()
            best_candidate0_index = -1
            best_candidate1_index = -1
            for permutation_index, permutation in enumerate(permutations):
                derive_offspring_0 = bool(random.getrandbits(1))
                candidate_offspring = copy.deepcopy(
                    offspring0) if derive_offspring_0 else copy.deepcopy(offspring1)
                for (item_index, parent_index) in permutation:
                    parent = parent0 if parent_index == 0 else parent1
                    candidate_offspring.add_item(
                        item_index, parent.placed_items[item_index].position, parent.placed_items[item_index].rotation)
                candidate_fitness = get_fitness(candidate_offspring)
                best_candidate_index = best_candidate0_index if derive_offspring_0 else best_candidate1_index
                if best_candidate_index == -1 or candidate_fitness > get_fitness(candidates[best_candidate_index]):
                    if derive_offspring_0:
                        best_candidate0_index = permutation_index
                    else:
                        best_candidate1_index = permutation_index
                if candidate_fitness >= min_fitness_for_non_best:
                    high_fitness_candidate_indices.append(permutation_index)
                candidates.append(candidate_offspring)

            # xác định các ứng viên tốt nhất và chọn chúng; nếu không có hoán vị nào tạo ra một phiên bản thay thế của cá thể con cơ bản (một trong số chúng hoặc cả hai), sử dụng các cá thể con cũ làm kết quả cuối cùng
            if best_candidate0_index != -1:
                offspring0 = candidates[best_candidate0_index]
            if best_candidate1_index != -1:
                offspring1 = candidates[best_candidate1_index]

            # nếu có các giải pháp thay thế không phải là tốt nhất với độ phù hợp cao, trả về chúng cùng với các giải pháp tốt nhất
            if high_fitness_candidate_indices:
                return [offspring0, offspring1] + [candidates[index] for index in high_fitness_candidate_indices if index != best_candidate0_index and index != best_candidate1_index]

    # trong trường hợp mặc định, chỉ trả về cặp cá thể con tốt nhất đã tạo ra
    return offspring0, offspring1


def mutate_with_addition(solution, max_attempt_num):
    """Cố gắng đột biến giải pháp đã cho bằng cách thêm một mục"""

    # tìm trọng lượng có thể thêm vào hộp chứa
    remaining_weight = solution.problem.container.max_weight - solution.weight

    # xác định các mục khả thi: những mục nằm trong giới hạn sức chứa và chưa được đặt
    feasible_item_indices = []
    for index, item in enumerate(solution.problem.items):
        if item.weight <= remaining_weight and index not in solution.placed_items:
            feasible_item_indices.append(index)

    # chỉ tiếp tục nếu có các mục khả thi
    if feasible_item_indices:

        # xác định các giới hạn của hộp chứa
        min_x, min_y, max_x, max_y = get_bounds(
            solution.problem.container.shape)

        # chọn ngẫu nhiên một mục
        item_index = random.choice(feasible_item_indices)

        # thực hiện tối đa một số lần thử
        for _ in range(max_attempt_num):

            # nếu hành động thành công, không cần thử thêm
            if solution.add_item(item_index, (random.uniform(min_x, max_x), random.uniform(min_y, max_y)), random.uniform(0, 360)):
                return True

    return False


def mutate_with_removal(solution):
    """Cố gắng đột biến giải pháp đã cho bằng cách loại bỏ một mục"""

    # loại bỏ chỉ có thể nếu có các mục đã được đặt trong hộp chứa
    if solution.placed_items:

        # chọn ngẫu nhiên một mục
        item_index = solution.get_random_placed_item_index()

        # loại bỏ mục
        return solution.remove_item(item_index)

    return False


def mutate_with_placement_modification(solution, max_attempt_num, small_position_change_proportion, small_rotation_change_proportion, move_until_intersection_point_num, move_until_intersection_min_dist_proportion, rotate_until_intersection_angle_num, item_index=-1, action_index=-1):
    """Cố gắng đột biến giải pháp đã cho bằng cách thay đổi một vị trí hiện có"""

    # các vị trí phải tồn tại để được thay đổi
    if solution.placed_items:

        # xác định các giới hạn và chiều dài cạnh của hộp chứa
        min_x, min_y, max_x, max_y = get_bounds(
            solution.problem.container.shape)
        x_length = abs(min_x - max_x)
        y_length = abs(min_y - max_y)

        # xác định sự dịch chuyển tối đa có thể được áp dụng trong các thay đổi vị trí nhỏ, dựa trên chiều dài cạnh của hộp chứa
        max_small_x_change = small_position_change_proportion * x_length
        max_small_y_change = small_position_change_proportion * y_length

        # xác định sự quay tối đa có thể được áp dụng trong các thay đổi quay nhỏ
        max_small_rotation = 360 * small_rotation_change_proportion

        # xác định khoảng cách tối thiểu và tối đa liên quan cho hành động di chuyển-cho đến khi giao nhau
        move_until_intersection_min_dist = np.linalg.norm(
            (x_length * move_until_intersection_min_dist_proportion, y_length * move_until_intersection_min_dist_proportion))
        move_until_intersection_max_dist = x_length + y_length

        # chọn một mục đã đặt ngẫu nhiên, nếu không được áp đặt từ bên ngoài
        if item_index < 0:
            item_index = solution.get_random_placed_item_index()

        # hoán đổi các thuộc tính giữa các mục chỉ có thể nếu có ít nhất hai vị trí
        can_swap = len(solution.placed_items) >= 2

        # liệt kê các hành động có thể
        position_change_index, small_position_change_index, move_until_intersect_index, position_swap_index, rotation_change_index, small_rotation_change_index, rotate_until_intersect_index, rotation_swap_index, position_rotation_change_index, small_position_rotation_change_index, position_rotation_swap_index = range(
            11)

        # nếu chỉ số hành động chưa được áp đặt từ bên ngoài, tính toán nó
        if action_index < 0:

            # một mục luôn có thể chọn thay đổi vị trí của nó
            action_indices = [
                position_change_index, small_position_change_index, move_until_intersect_index]
            if can_swap:
                action_indices.append(position_swap_index)

            # chỉ một số mục có thể được quay
            if not np.isnan(solution.placed_items[item_index].rotation):
                action_indices.extend([rotation_change_index, small_rotation_change_index, rotate_until_intersect_index,
                                      position_rotation_change_index, small_position_rotation_change_index])
                if can_swap:
                    action_indices.extend(
                        [rotation_swap_index, position_rotation_swap_index])

            # chọn một hành động thay đổi, với cùng xác suất cho tất cả các hành động có sẵn
            action_index = random.choice(action_indices)

        # lưu trữ các chỉ số của các mục mà không nên thử hoán đổi, để mở rộng khi thất bại
        swap_ignore_indices = [item_index]

        # thực hiện tối đa một số lần thử
        for _ in range(max_attempt_num):

            has_modified = True

            # cố gắng áp dụng hành động
            if action_index == position_change_index:
                has_modified = solution.move_item_to(
                    item_index, (random.uniform(min_x, max_x), random.uniform(min_y, max_y)))
            elif action_index == small_position_change_index:
                has_modified = solution.move_item(item_index, (random.uniform(
                    -max_small_x_change, max_small_x_change), random.uniform(-max_small_y_change, max_small_y_change)))
            elif action_index == move_until_intersect_index:
                has_modified = solution.move_item_in_direction(item_index, (random.uniform(-1, 1), random.uniform(-1, 1)),
                                                               move_until_intersection_point_num, move_until_intersection_min_dist, move_until_intersection_max_dist)
            elif action_index == position_swap_index:
                swap_item_index = solution.get_random_placed_item_index(
                    swap_ignore_indices)
                if not swap_item_index:
                    break
                has_modified = solution.swap_placements(
                    item_index, swap_item_index, swap_position=True, swap_rotation=False)
                swap_ignore_indices.append(swap_item_index)
            elif action_index == rotation_change_index:
                has_modified = solution.rotate_item_to(
                    item_index, random.uniform(0, 360))
            elif action_index == small_rotation_change_index:
                has_modified = solution.rotate_item(
                    item_index, random.uniform(-max_small_rotation, max_small_rotation))
            elif action_index == rotate_until_intersect_index:
                clockwise = bool(random.getrandbits(1))
                has_modified = solution.rotate_item_in_direction(
                    item_index, clockwise, rotate_until_intersection_angle_num)
                if not has_modified:
                    has_modified = solution.rotate_item_in_direction(
                        item_index, not clockwise, rotate_until_intersection_angle_num)
            elif action_index == rotation_swap_index:
                swap_item_index = solution.get_random_placed_item_index(
                    swap_ignore_indices)
                if not swap_item_index:
                    break
                has_modified = solution.swap_placements(
                    item_index, swap_item_index, swap_position=False, swap_rotation=True)
                swap_ignore_indices.append(swap_item_index)
            elif action_index == position_rotation_change_index:
                has_modified = solution.move_and_rotate_item_to(item_index, (random.uniform(
                    min_x, max_x), random.uniform(min_y, max_y)), random.uniform(0, 360))
            elif action_index == small_position_rotation_change_index:
                has_modified = solution.move_and_rotate_item(item_index, (random.uniform(-max_small_x_change, max_small_x_change),
                                                             random.uniform(-max_small_y_change, max_small_y_change)), random.uniform(-max_small_rotation, max_small_rotation))
            elif action_index == position_rotation_swap_index:
                swap_item_index = solution.get_random_placed_item_index(
                    swap_ignore_indices)
                if not swap_item_index:
                    break
                has_modified = solution.swap_placements(
                    item_index, swap_item_index, swap_position=True, swap_rotation=True)
                swap_ignore_indices.append(swap_item_index)

            # nếu hành động được xác nhận thành công, không cần thử thêm
            if has_modified:
                return True

            # nếu hai hướng có thể cho hành động quay-cho đến khi giao nhau đã được thử (không thành công) cho mục, không còn gì để làm
            if action_index == rotate_until_intersect_index:
                return False

    return False


def get_mutation(solution, min_iter_num, mutation_add_weight, mutation_remove_weight, mutation_modify_weight, mutation_add_max_attempt_num, mutation_modify_max_attempt_num, small_position_change_proportion, small_rotation_change_proportion,  mutation_modify_move_until_intersection_point_num, mutation_modify_move_until_intersection_min_dist_proportion, mutation_modify_rotate_until_intersection_angle_num, mutation_intermediate_selection_prob):
    """Tạo và trả về một bản sao đột biến của giải pháp đã cho"""

    # tất cả các thay đổi sẽ được áp dụng cho một bản sao của cá thể
    mutated_solution = copy.deepcopy(solution)

    # tạm thời giữ lại giải pháp gốc như là giải pháp tốt nhất
    best_solution = solution

    removal_num_to_compensate = 0
    iter_count = 0

    action_indices = [add_index, remove_index, _] = [0, 1, 2]
    action_weights = [mutation_add_weight,
                      mutation_remove_weight, mutation_modify_weight]

    # thực hiện các vòng lặp ít nhất một số lần tối thiểu và tiếp tục nếu cần bù đắp các lần loại bỏ
    while iter_count < min_iter_num or removal_num_to_compensate > 0:

        # sau số vòng lặp tối thiểu, thêm các mục để bù đắp các lần loại bỏ
        if iter_count >= min_iter_num:
            action_index = add_index

        # nếu không, thực hiện một lựa chọn có trọng số của hành động tiếp theo cần làm
        else:
            action_index = random.choices(
                action_indices, action_weights, k=1)[0]

        # đột biến với loại hành động đã chọn
        if action_index == add_index:
            has_mutated = mutate_with_addition(
                mutated_solution, mutation_add_max_attempt_num)
            # nếu việc thêm thành công, nó có thể bù đắp một lần loại bỏ trước đó; cũng tính là đã bù đắp bất kỳ lần thêm thất bại nào sau giới hạn vòng lặp cơ bản, để tránh một vòng lặp vô hạn
            if has_mutated or iter_count >= min_iter_num:
                removal_num_to_compensate = max(
                    removal_num_to_compensate - 1, 0)
        elif action_index == remove_index:
            has_mutated = mutate_with_removal(mutated_solution)
            # nếu việc loại bỏ thành công, nó sẽ cần được bù đắp
            if has_mutated:
                removal_num_to_compensate += 1
        else:
            has_mutated = mutate_with_placement_modification(mutated_solution, mutation_modify_max_attempt_num, small_position_change_proportion, small_rotation_change_proportion,
                                                             mutation_modify_move_until_intersection_point_num, mutation_modify_move_until_intersection_min_dist_proportion, mutation_modify_rotate_until_intersection_angle_num)

        # nếu một đột biến đã được áp dụng và có bất kỳ khả năng nào để chọn các giải pháp trung gian làm giải pháp cuối cùng và giải pháp trung gian hiện tại tốt hơn bất kỳ giải pháp trước đó nào, giữ lại một bản sao của nó
        if has_mutated and mutation_intermediate_selection_prob > 0 and get_fitness(mutated_solution) > get_fitness(best_solution):
            best_solution = copy.deepcopy(mutated_solution)

        iter_count += 1

    # nếu có thể, chọn một giải pháp trung gian làm giải pháp cuối cùng với một xác suất nhất định nếu nó tốt hơn giải pháp đột biến cuối cùng
    if mutation_intermediate_selection_prob > 0 and mutated_solution != best_solution and best_solution != solution and random.uniform(0, 1) < mutation_intermediate_selection_prob:
        return best_solution
    # nếu không, giữ lại giải pháp đột biến cuối cùng làm giải pháp cuối cùng
    else:
        return mutated_solution


def generate_offspring(parents, mutation_min_iter_num, mutation_add_weight, mutation_remove_weight, mutation_modify_weight, mutation_add_max_attempt_num, mutation_modify_max_attempt_num, small_position_change_proportion, small_rotation_change_proportion, mutation_modify_move_until_intersection_point_num, mutation_modify_move_until_intersection_min_dist_proportion, mutation_modify_rotate_until_intersection_angle_num, mutation_intermediate_selection_prob, crossover_ignore_mutation_probability, crossover_max_attempt_num, crossover_shape_min_length_proportion, crossover_shape_max_length_proportion, crossover_shape_min_area_proportion, crossover_polygon_max_vertex_num, crossover_max_permutation_num, crossover_min_fitness_for_non_best, calculate_times=False):
    """Tạo và trả về các cá thể con từ các cha mẹ đã cho"""

    start_time = crossover_time = mutation_time = 0

    # các cha mẹ cần tồn tại để tạo ra cá thể con
    if not parents:
        return list()

    # nếu các cha mẹ được biểu diễn dưới dạng cặp, phép lai tạo nên được áp dụng trước khi đột biến
    can_use_crossover = type(parents[0]) == tuple

    # nếu không có phép lai tạo, các cha mẹ sẽ là cơ sở cho các đột biến
    individuals_to_mutate = parents if not can_use_crossover else list()

    # nếu có thể, thực hiện phép lai tạo
    if can_use_crossover:

        if calculate_times:
            start_time = time.time()

        # sử dụng phép lai tạo để tạo ra (ít nhất) hai cá thể cho mỗi cặp, sẽ được đột biến sau đó
        for (parent0, parent1) in parents:
            individuals_to_mutate.extend(get_crossover(parent0, parent1, crossover_max_attempt_num, crossover_shape_min_length_proportion, crossover_shape_max_length_proportion,
                                         crossover_shape_min_area_proportion, crossover_polygon_max_vertex_num, crossover_max_permutation_num, crossover_min_fitness_for_non_best))

        if calculate_times:
            crossover_time += get_time_since(start_time)

    offspring = list()

    if calculate_times:
        start_time = time.time()

    # đột biến các cá thể để có được cá thể con cuối cùng
    for individual in individuals_to_mutate:

        # nếu cá thể đã được tạo ra bằng phép lai tạo, bỏ qua đột biến hoàn toàn với một xác suất nhất định
        if can_use_crossover and random.uniform(0, 1) < crossover_ignore_mutation_probability:
            offspring.append(individual)

        else:
            # tạo một bản sao đột biến của cá thể
            mutated_individual = get_mutation(individual, mutation_min_iter_num, mutation_add_weight, mutation_remove_weight, mutation_modify_weight, mutation_add_max_attempt_num, mutation_modify_max_attempt_num, small_position_change_proportion,
                                              small_rotation_change_proportion,  mutation_modify_move_until_intersection_point_num, mutation_modify_move_until_intersection_min_dist_proportion, mutation_modify_rotate_until_intersection_angle_num, mutation_intermediate_selection_prob)

            # nếu phép lai tạo đã được sử dụng và cá thể đột biến ít phù hợp hơn (hoặc cùng mức nhưng thua trong phá vỡ hòa), giữ lại cá thể trước đột biến với một xác suất nhất định
            # if can_use_crossover and get_fittest_solution([individual, mutated_individual]) == individual:
            if can_use_crossover and get_fittest_solution_with_circle([individual, mutated_individual]) == individual:
                offspring.append(individual)

            # trong điều kiện bình thường, giữ lại cá thể đột biến
            else:
                offspring.append(mutated_individual)

    if calculate_times:
        mutation_time += get_time_since(start_time)

    if calculate_times:
        return offspring, crossover_time, mutation_time

    return offspring


def get_surviving_population(population, survivor_num, population_update_pool_size):
    """Sử dụng lựa chọn đấu loại (với các nhóm có kích thước đã cho) để khôi phục một quần thể với số lượng cá thể sống sót đã cho"""

    # nếu số lượng cá thể sống sót bằng hoặc lớn hơn kích thước quần thể hiện tại, trả về toàn bộ quần thể
    if survivor_num >= len(population):
        return population

    surviving_population = list()

    # nếu không, chọn số lượng cá thể sống sót cần thiết
    for _ in range(survivor_num):

        # một người chiến thắng trong đấu loại có thể sống sót
        survivor = get_tournament_winner(
            population, population_update_pool_size)

        # thêm cá thể sống sót vào quần thể mới
        surviving_population.append(survivor)

        # loại bỏ cá thể sống sót khỏi quần thể cũ, để tránh trùng lặp
        population.remove(survivor)

    return surviving_population


def sort_by_fitness(population):
    """Sắp xếp quần thể đã cho theo độ phù hợp giảm dần, tại chỗ, mà không xem xét các tiêu chí phá vỡ hòa"""

    population.sort(key=lambda solution: get_fitness(solution), reverse=True)


def get_fitness_stats(population):
    """Trả về một từ điển với thông tin thống kê về độ phù hợp của quần thể đã cho"""

    max_fitness = -np.inf
    min_fitness = np.inf
    fitness_sum = 0
    fitness_counts = dict()

    for solution in population:
        fitness = get_fitness(solution)
        fitness_sum += fitness
        if fitness > max_fitness:
            max_fitness = fitness
        if fitness < min_fitness:
            min_fitness = fitness
        if fitness in fitness_counts:
            fitness_counts[fitness] += 1
        else:
            fitness_counts[fitness] = 0

    return {"max": max_fitness, "min": min_fitness, "avg": fitness_sum / len(population), "mode": max(fitness_counts, key=fitness_counts.get)}


def solve_problem(problem, population_size=POPULATION_SIZE, initial_generation_item_specialization_iter_proportion=INITIAL_SOLUTION_GENERATION_FIRST_ITEM_SPECIALIZATION_ITER_PROPORTION, offspring_size=OFFSPRING_SIZE, elite_size=ELITE_SIZE, parent_selection_pool_size=PARENT_SELECTION_POOL_SIZE, population_update_pool_size=POPULATION_UPDATE_POOL_SIZE, max_generation_num=MAX_GENERATION_NUM, converge_generation_num=CONVERGE_GENERATION_NUM, mutation_min_iter_num=MUTATION_MIN_ITER_NUM, mutation_add_weight=MUTATION_ADD_WEIGHT, mutation_remove_weight=MUTATION_REMOVE_WEIGHT, mutation_modify_weight=MUTATION_MODIFY_WEIGHT, mutation_add_max_attempt_num=MUTATION_ADD_MAX_ATTEMPT_NUM, mutation_modify_max_attempt_num=MUTATION_MODIFY_MAX_ATTEMPT_NUM, small_position_change_proportion=MUTATION_MODIFY_SMALL_POSITION_CHANGE_PROPORTION, small_rotation_change_proportion=MUTATION_MODIFY_SMALL_ROTATION_CHANGE_PROPORTION, mutation_modify_move_until_intersection_point_num=MUTATION_MODIFY_MOVE_UNTIL_INTERSECTION_POINT_NUM, mutation_modify_move_until_intersection_min_dist_proportion=MUTATION_MODIFY_MOVE_UNTIL_INTERSECTION_MIN_DIST_PROPORTION, mutation_modify_rotate_until_intersection_angle_num=MUTATION_MODIFY_ROTATE_UNTIL_INTERSECTION_ANGLE_NUM, mutation_intermediate_selection_prob=MUTATION_INTERMEDIATE_SELECTION_PROB, can_use_crossover=CAN_USE_CROSSOVER, crossover_ignore_mutation_probability=CROSSOVER_IGNORE_MUTATION_PROBABILITY, crossover_max_attempt_num=CROSSOVER_MAX_ATTEMPT_NUM, crossover_shape_min_length_proportion=CROSSOVER_SHAPE_MIN_LENGTH_PROPORTION, crossover_shape_max_length_proportion=CROSSOVER_SHAPE_MAX_LENGTH_PROPORTION, crossover_shape_min_area_proportion=CROSSOVER_SHAPE_MIN_AREA_PROPORTION, crossover_polygon_max_vertex_num=CROSSOVER_POLYGON_MAX_VERTEX_NUM, crossover_max_permutation_num=CROSSOVER_MAX_PERMUTATION_NUM, crossover_min_fitness_for_non_best_proportion=CROSSOVER_MIN_FITNESS_FOR_NON_BEST_PROPORTION, calculate_times=False, return_population_fitness_per_generation=False):
    """Tìm và trả về một giải pháp cho vấn đề đã cho, sử dụng thuật toán tiến hóa"""

    start_time = initial_population_time = parent_selection_time = min_fitness_for_crossover_non_best_time = crossover_time = mutation_time = elite_finding_time = population_update_time = early_stop_time = fitness_per_generation_time = 0

    # một giá trị đại diện cho sự khác biệt tối thiểu của giá trị hộp chứa cho vấn đề, được tìm thấy bằng cách giảm giá trị thấp nhất trong số các mục
    minimal_value_difference = min(
        [item.value for item in problem.items]) * 0.00001

    if calculate_times:
        start_time = time.time()

    # tạo ra quần thể ban đầu
    population = generate_population(
        problem, population_size, initial_generation_item_specialization_iter_proportion)

    if calculate_times:
        initial_population_time += get_time_since(start_time)

    if return_population_fitness_per_generation:
        population_fitness_per_generation = list()
    else:
        population_fitness_per_generation = None

    max_fitness = -np.inf
    iter_count_without_improvement = 0
    extended_population = None

    elite = list()

    # cập nhật quần thể lên đến một số thế hệ tối đa
    for i in range(max_generation_num):

        # nếu cần, lưu trữ độ phù hợp của từng cá thể trong quần thể của thế hệ hiện tại
        if return_population_fitness_per_generation:

            if calculate_times:
                start_time = time.time()

            population_fitness_per_generation.append(
                [get_fitness(solution) for solution in population])

            if calculate_times:
                fitness_per_generation_time += get_time_since(start_time)

        if calculate_times:
            start_time = time.time()

        # chọn các cá thể làm cha mẹ để tạo ra con cái
        parents = select_parents(
            population, offspring_size, parent_selection_pool_size, can_use_crossover)

        if calculate_times:
            parent_selection_time += get_time_since(start_time)

        if calculate_times:
            start_time = time.time()

        # tính toán độ phù hợp tối thiểu cho các giải pháp không phải là tốt nhất (thay thế) trong phép lai tạo để được chấp nhận như là con cái bổ sung; nó đơn giản để tìm ngưỡng khi quần thể được sắp xếp theo thứ tự giảm dần của độ phù hợp, và một sự cải thiện về độ phù hợp (dù là tối thiểu) từ một cá thể tham chiếu (theo tỷ lệ) cần phải xảy ra
        crossover_min_fitness_for_non_best = 0
        if can_use_crossover and crossover_min_fitness_for_non_best_proportion > 0:
            need_to_improve_best = crossover_min_fitness_for_non_best_proportion >= 1
            sort_by_fitness(population)
            min_fitness_index = 0 if need_to_improve_best else math.ceil(
                (1. - crossover_min_fitness_for_non_best_proportion) * len(population))
            crossover_min_fitness_for_non_best = get_fitness(
                population[min_fitness_index])
            crossover_min_fitness_for_non_best += minimal_value_difference

        if calculate_times:
            min_fitness_for_crossover_non_best_time += get_time_since(
                start_time)

        # tạo ra con cái
        offspring_result = generate_offspring(parents, mutation_min_iter_num, mutation_add_weight, mutation_remove_weight, mutation_modify_weight, mutation_add_max_attempt_num, mutation_modify_max_attempt_num, small_position_change_proportion, small_rotation_change_proportion, mutation_modify_move_until_intersection_point_num, mutation_modify_move_until_intersection_min_dist_proportion,
                                              mutation_modify_rotate_until_intersection_angle_num, mutation_intermediate_selection_prob, crossover_ignore_mutation_probability, crossover_max_attempt_num, crossover_shape_min_length_proportion, crossover_shape_max_length_proportion, crossover_shape_min_area_proportion, crossover_polygon_max_vertex_num, crossover_max_permutation_num, crossover_min_fitness_for_non_best, calculate_times)
        if calculate_times:
            offspring, new_crossover_time, new_mutation_time = offspring_result
            crossover_time += new_crossover_time
            mutation_time += new_mutation_time
        else:
            offspring = offspring_result

        if calculate_times:
            start_time = time.time()

        # xác định một quần thể mở rộng tạm thời bằng cách kết hợp quần thể gốc và con cái của chúng
        extended_population = population + offspring

        # tìm các cá thể ưu tú (những cá thể có độ phù hợp cao nhất) trong quần thể mở rộng, được sắp xếp theo thứ tự giảm dần của độ phù hợp
        elite = get_fittest_solutions(extended_population, elite_size)

        if calculate_times:
            elite_finding_time += get_time_since(start_time)

        if calculate_times:
            start_time = time.time()

        # kiểm tra xem độ phù hợp có cải thiện trong thế hệ này không, để loại bỏ hoặc đóng góp vào việc xác nhận giả định hội tụ
        elite_fitness = get_fitness(elite[0]) if elite else -np.inf
        if elite_fitness > max_fitness:
            max_fitness = elite_fitness
            iter_count_without_improvement = 0
        else:
            iter_count_without_improvement += 1

        # dừng sớm nếu độ phù hợp của cá thể ưu tú không cải thiện trong một số vòng lặp nhất định
        if iter_count_without_improvement >= converge_generation_num:
            break

        if calculate_times:
            early_stop_time += get_time_since(start_time)

        if calculate_times:
            start_time = time.time()

        # cập nhật quần thể, dành chỗ cho các cá thể ưu tú và chọn phần còn lại từ quần thể mở rộng, để khôi phục kích thước quần thể tiêu chuẩn
        population = elite + get_surviving_population([individual for individual in extended_population if individual not in elite],
                                                      population_size - len(elite), population_update_pool_size)

        if calculate_times:
            population_update_time += get_time_since(start_time)

    # giải pháp có độ phù hợp cao nhất là giải pháp tốt nhất
    # best_solution = elite[0] if elite else get_fittest_solution(population)
    best_solution = elite[0] if elite else get_fittest_solution_with_circle(
        population)

    # nếu cần, lưu trữ độ phù hợp của từng cá thể trong quần thể cuối cùng (nếu dừng sớm, trước khi chọn cá thể sống sót, chọn chúng ngay bây giờ)
    if return_population_fitness_per_generation:

        if calculate_times:
            start_time = time.time()

        if iter_count_without_improvement >= converge_generation_num:
            population = elite + get_surviving_population([individual for individual in extended_population if individual not in elite],
                                                          population_size - len(elite), population_update_pool_size)

        population_fitness_per_generation.append(
            [get_fitness(solution) for solution in population])

        if calculate_times:
            fitness_per_generation_time += get_time_since(start_time)

    # đóng gói tất cả các thời gian một cách thông tin trong một từ điển
    if calculate_times:
        approx_total_time = initial_population_time + parent_selection_time + min_fitness_for_crossover_non_best_time + \
            crossover_time + mutation_time + elite_finding_time + \
            population_update_time + early_stop_time + fitness_per_generation_time
        time_dict = {"Generation of the initial population": (initial_population_time, initial_population_time / approx_total_time), "Parent selection": (parent_selection_time, parent_selection_time / approx_total_time), "Min-fitness-for-crossover-non-best calculation": (min_fitness_for_crossover_non_best_time, min_fitness_for_crossover_non_best_time / approx_total_time), "Crossover": (crossover_time, crossover_time / approx_total_time), "Mutation": (
            mutation_time, mutation_time / approx_total_time), "Elite finding": (elite_finding_time, elite_finding_time / approx_total_time), "Population update": (population_update_time, population_update_time / approx_total_time), "Early stopping check": (early_stop_time, early_stop_time / approx_total_time), "Population fitness per generation gathering": (fitness_per_generation_time, fitness_per_generation_time / approx_total_time)}
        if return_population_fitness_per_generation:
            return best_solution, time_dict, population_fitness_per_generation
        return best_solution, time_dict

    if return_population_fitness_per_generation:
        return best_solution, population_fitness_per_generation

    return best_solution
