import copy
import itertools
import random
import time
from common_algo_functions import get_index_after_weight_limit, get_time_since
from solution import Solution
from utils import *

# trọng số mặc định của giá trị của một mục trong tổng trọng số của giá trị và tỷ lệ lợi nhuận, với trọng số của tỷ lệ lợi nhuận là 1-VALUE_WEIGHT
VALUE_WEIGHT = 0.5

# trọng số mặc định của diện tích trong tích trọng số của trọng lượng và diện tích của mục; trọng số của trọng lượng là 1-AREA_WEIGHT
AREA_WEIGHT = 0.5

# số vòng lặp tối đa mặc định cho thuật toán tham lam
MAX_ITER_NUM = 100000

# số vòng lặp tối đa mặc định không có sự thay đổi để thực hiện dừng sớm
MAX_ITER_NUM_WITHOUT_CHANGES = 30000

# số lần lặp lại mặc định của thuật toán
REPETITION_NUM = 1

# có cho phép sử dụng điểm số hằng số nếu được chỉ định rõ hay không
CAN_USE_CONSTANT_SCORE = True


def get_constant_score(item, value_weight, area_weight, can_use_constant_score=CAN_USE_CONSTANT_SCORE):
    """Trả về một điểm số hằng số bất kể các đặc tính của mục hay bất kỳ trọng số nào, nếu được phép"""

    if can_use_constant_score:
        return 1

    return get_weighted_sum_of_item_value_and_profitability_ratio(item, value_weight, area_weight)


def get_item_profitability_ratio(item, area_weight):
    """Trả về tỷ lệ lợi nhuận của một mục"""

    # tỷ lệ là giá trị của mục chia cho tích trọng số của trọng lượng và diện tích của hình dạng mục
    return item.value / ((1. - area_weight) * item.weight + area_weight * item.shape.area)


def get_weighted_sum_of_item_value_and_profitability_ratio(item, value_weight, area_weight):
    """Trả về tổng trọng số của giá trị và tỷ lệ lợi nhuận của một mục"""

    return value_weight * item.value + (1. - value_weight) * get_item_profitability_ratio(item, area_weight)


def select_item(items_with_profit_ratio):
    """Cho một danh sách các tuple dưới dạng (chỉ mục của mục, tỷ lệ lợi nhuận của mục, mục), chọn một mục theo tỷ lệ lợi nhuận của nó và trả về chỉ mục của nó"""

    # tìm tổng tỷ lệ lợi nhuận, mã dựa trên random.choices() từ thư viện chuẩn của Python
    cumulative_profit_ratios = list(itertools.accumulate(item_with_profit_ratio[1] for item_with_profit_ratio in items_with_profit_ratio))
    profit_ratio_sum = cumulative_profit_ratios[-1]

    # chọn ngẫu nhiên một tỷ lệ trong phạm vi tổng giá trị
    profit_ratio = random.uniform(0, profit_ratio_sum)

    # tìm giá trị nằm trong tỷ lệ ngẫu nhiên đã chọn; mã tìm kiếm nhị phân dựa trên bisect.bisect_left từ thư viện Python chuẩn, nhưng được điều chỉnh cho kiểm tra tỷ lệ lợi nhuận
    lowest = 0
    highest = len(items_with_profit_ratio)
    while lowest < highest:
        middle = (lowest + highest) // 2
        if cumulative_profit_ratios[middle] <= profit_ratio:
            lowest = middle + 1
        else:
            highest = middle

    return lowest


def solve_problem(problem, greedy_score_function=get_weighted_sum_of_item_value_and_profitability_ratio, value_weight=VALUE_WEIGHT, area_weight=AREA_WEIGHT, max_iter_num=MAX_ITER_NUM, max_iter_num_without_changes=MAX_ITER_NUM_WITHOUT_CHANGES, repetition_num=REPETITION_NUM, item_index_to_place_first=-1, item_specialization_iter_proportion=0., calculate_times=False, return_value_evolution=False):
    """Tìm và trả về giải pháp cho vấn đề đã đưa ra, sử dụng chiến lược tham lam"""

    # xác định giới hạn của hộp chứa
    min_x, min_y, max_x, max_y = get_bounds(problem.container.shape)

    max_item_specialization_iter_num = item_specialization_iter_proportion * max_iter_num

    start_time = 0
    sort_time = 0
    item_discarding_time = 0
    item_selection_time = 0
    addition_time = 0
    value_evolution_time = 0

    if calculate_times:
        start_time = time.time()

    if return_value_evolution:
        value_evolution = list()
    else:
        value_evolution = None

    if calculate_times:
        value_evolution_time += get_time_since(start_time)

    if calculate_times:
        start_time = time.time()

    # sắp xếp các mục (với điểm tham lam được tính) theo trọng lượng, để tăng tốc độ loại bỏ các mục khi chúng làm vượt quá dung lượng của hộp chứa
    original_items_by_weight = []
    for index, item in enumerate(sorted(problem.items, key=lambda item: item.weight)):
        original_items_by_weight.append((index, greedy_score_function(item, value_weight, area_weight), item))

    if calculate_times:
        sort_time += get_time_since(start_time)

    if calculate_times:
        start_time = time.time()

    # loại bỏ các mục sẽ làm vượt quá dung lượng của hộp chứa
    original_items_by_weight = original_items_by_weight[:get_index_after_weight_limit(original_items_by_weight, problem.container.max_weight)]

    if calculate_times:
        item_discarding_time += get_time_since(start_time)

    best_solution = None

    # nếu thuật toán được lặp lại, nó sẽ được lặp lại và giải pháp tốt nhất được giữ lại ở cuối cùng
    for _ in range(repetition_num):

        # nếu thuật toán được lặp lại, sử dụng một bản sao của các mục đã sắp xếp ban đầu để bắt đầu lại từ đầu lần sau
        if repetition_num > 1:
            items_by_weight = copy.deepcopy(original_items_by_weight)
        else:
            items_by_weight = original_items_by_weight

        # tạo một giải pháp ban đầu không có mục nào được đặt trong hộp chứa
        solution = Solution(problem)

        # việc đặt mục chỉ có thể thực hiện được với dung lượng và các mục hợp lệ
        if problem.container.max_weight and items_by_weight:

            iter_count_without_changes = 0

            # cố gắng thêm các mục vào hộp chứa, với số vòng lặp tối đa
            for i in range(max_iter_num):

                if calculate_times:
                    start_time = time.time()

                # nếu cần, chọn một mục cụ thể để cố gắng đặt (chỉ trong một số lần cố gắng tối đa)
                if item_index_to_place_first >= 0 and i < max_item_specialization_iter_num:
                    item_index = item_index_to_place_first
                    list_index = -1

                # thực hiện chọn ngẫu nhiên mục tiếp theo để cố gắng đặt, mỗi mục được trọng số với tỷ lệ lợi nhuận của chúng, hoạt động như một xác suất lựa chọn ngẫu nhiên
                else:
                    list_index = select_item(items_by_weight)
                    item_index = items_by_weight[list_index][0]

                if calculate_times:
                    item_selection_time += get_time_since(start_time)

                if calculate_times:
                    start_time = time.time()

                # cố gắng thêm mục vào một vị trí ngẫu nhiên và với một góc quay ngẫu nhiên; nếu hợp lệ, xóa mục đó khỏi danh sách mục đang chờ
                if solution.add_item(item_index, (random.uniform(min_x, max_x), random.uniform(min_y, max_y)), random.uniform(0, 360)):

                    # mục để đặt trước được coi là đã được đặt, nếu có
                    item_index_to_place_first = -1

                    if calculate_times:
                        addition_time += get_time_since(start_time)

                    # tìm trọng lượng còn lại có thể thêm vào
                    remaining_weight = problem.container.max_weight - solution.weight

                    # dừng sớm nếu dung lượng đã được đạt chính xác
                    if not remaining_weight:
                        break

                    # xóa mục đã được đặt khỏi danh sách mục đang chờ
                    if list_index >= 0:
                        items_by_weight.pop(list_index)

                    # nếu tập trung vào một mục để đặt trước, tìm mục đó trong danh sách để xóa nó
                    else:
                        for list_i in range(len(items_by_weight)):
                            if items_by_weight[list_i][0] == item_index:
                                items_by_weight.pop(list_i)
                                break

                    if calculate_times:
                        start_time = time.time()

                    # loại bỏ các mục sẽ làm vượt quá dung lượng còn lại
                    items_by_weight = items_by_weight[:get_index_after_weight_limit(items_by_weight, remaining_weight)]

                    if calculate_times:
                        item_discarding_time += get_time_since(start_time)

                    # dừng sớm nếu không thể thêm nhiều mục hơn, vì tất cả đã được đặt hoặc tất cả các mục bên ngoài sẽ làm vượt quá dung lượng
                    if not items_by_weight:
                        break

                    # đặt lại bộ đếm hội tụ tiềm năng, vì một mục đã được thêm vào
                    iter_count_without_changes = 0

                else:

                    if calculate_times:
                        addition_time += get_time_since(start_time)

                    # ghi nhận việc không thể đặt một mục trong vòng lặp này
                    iter_count_without_changes += 1

                    # dừng sớm nếu có quá nhiều vòng lặp không có thay đổi (trừ khi một mục cụ thể đang được cố gắng đặt trước)
                    if iter_count_without_changes >= max_iter_num_without_changes and item_index_to_place_first < 0:
                        break

                if return_value_evolution:

                    if calculate_times:
                        start_time = time.time()

                    value_evolution.append(solution.value)

                    if calculate_times:
                        value_evolution_time += get_time_since(start_time)

        # nếu thuật toán sử dụng nhiều vòng lặp, chấp nhận giải pháp hiện tại làm giải pháp tốt nhất nếu nó có giá trị cao nhất cho đến thời điểm hiện tại
        if not best_solution or solution.value > best_solution.value:
            best_solution = solution

    # encapsulate all times informatively in a dictionary
    if calculate_times:
        approx_total_time = sort_time + item_selection_time + item_discarding_time + addition_time + value_evolution_time
        time_dict = {"Sắp xếp theo trọng lượng và tính toán tỷ lệ lợi nhuận": (sort_time, sort_time / approx_total_time), "Lựa chọn mục ngẫu nhiên": (item_selection_time, item_selection_time / approx_total_time), "Loại bỏ mục": (
            item_discarding_time, item_discarding_time / approx_total_time), "Thêm và xác nhận hình học": (addition_time, addition_time / approx_total_time), "Lưu giữ giá trị của mỗi vòng lặp": (value_evolution_time, value_evolution_time / approx_total_time)}
        if return_value_evolution:
            return best_solution, time_dict, value_evolution
        return best_solution, time_dict

    if return_value_evolution:
        return best_solution, value_evolution

    return best_solution
