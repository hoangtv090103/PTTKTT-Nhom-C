import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

# figure size for plots
PLOT_FIG_SIZE = (16, 9)

# font size for plot titles
PLOT_TITLE_SIZE = 16

# DPI for plots
PLOT_DPI = 200


def get_bounds(shape):
    """Trả về giới hạn (min_x, min_y, max_x, max_y) của một hình dạng"""
    if hasattr(shape, 'bounds'):
        bounds = shape.bounds
    elif hasattr(shape, 'to_polygon'):
        # Nếu hình không có bounds trực tiếp nhưng có thể chuyển đổi thành polygon
        bounds = shape.to_polygon().bounds
    else:
        raise AttributeError(f"Shape of type {type(shape)} has no bounds attribute or to_polygon method")
    return bounds[0], bounds[1], bounds[2], bounds[3]


def get_stats(values, round_digits=None):
    """Tính toán các thống kê cơ bản của một danh sách giá trị"""
    if not values:
        return 0, 0, 0, 0, 0

    mean = np.mean(values)
    std = np.std(values)
    min_ = min(values)
    median = np.median(values)
    max_ = max(values)

    if round_digits is not None:
        mean = round(mean, round_digits)
        std = round(std, round_digits)
        min_ = round(min_, round_digits)
        median = round(median, round_digits)
        max_ = round(max_, round_digits)

    return mean, std, min_, median, max_


def print_if_allowed(value, can_print):
    """In giá trị nếu được phép"""
    if can_print:
        print(value)
    return value


def add_newlines_by_spaces(string, line_length):
    """Return a version of the passed string where spaces (or hyphens) get replaced with a new line if the accumulated number of characters has exceeded the specified line length"""

    replacement_indices = set()
    current_length = 0

    for i, char in enumerate(string):
        current_length += 1
        if current_length > line_length and (char == " " or char == "-"):
            replacement_indices.add(i)
            current_length = 0

    return "".join(
        [("\n" if i in replacement_indices else char) for i, char in enumerate(string)]
    )


def get_time_since(start_time):
    """Tính thời gian đã trôi qua từ thời điểm bắt đầu (milliseconds)"""
    return (time.time() - start_time) * 1000


def get_index_after_weight_limit(items_by_weight, weight_limit):
    """Given a list of (item_index, [any other fields,] item) tuples sorted by item weight, find the positional index that first exceeds the passed weight limit"""

    # find the item that first exceeds the weight limit; binary search code is based on bisect.bisect_right from standard Python library, but adapted to weight check
    lowest = 0
    highest = len(items_by_weight)
    while lowest < highest:
        middle = (lowest + highest) // 2
        if items_by_weight[middle][-1].weight >= weight_limit:
            highest = middle
        else:
            lowest = middle + 1
    return lowest


def visualize_plot(values, title, labels=None, show_plot=True, save_path=None):
    """Given a list of sequential values, show a line-based visualization (and/or save it to a file, as indicated in the parameters)"""

    fig, ax = plt.subplots(figsize=PLOT_FIG_SIZE)
    ax.set_title(title, fontsize=PLOT_TITLE_SIZE)

    if labels is None:
        labels = list(range(len(values)))

    plt.plot(labels, list(values.values()))

    fig = plt.gcf()

    if show_plot:
        plt.show()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=PLOT_DPI)
        plt.close(fig)


def visualize_boxplot_for_data_sequence(
    data_lists,
    title,
    labels=None,
    y_scale_override=None,
    show_plot=True,
    save_path=None,
):
    """Given a list of lists (assumed to collectively define a sequence), a boxplot is shown (and/or save it to a file) for the data values of each sublist"""

    fig, ax = plt.subplots(figsize=PLOT_FIG_SIZE)
    ax.set_title(title, fontsize=PLOT_TITLE_SIZE)

    if not labels:
        labels = range(len(data_lists))

    ax.boxplot(
        data_lists,
        labels=labels,
        showmeans=True,
        meanline=True,
        whis=[5, 95],
        flierprops=dict(markerfacecolor="black", marker="+"),
    )

    # based on: https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/custom_legends.html
    legend_lines = [
        Line2D([0], [0], color="orange", lw=1, ls="-"),
        Line2D([0], [0], color="g", lw=1, ls="--"),
    ]
    ax.legend(
        legend_lines,
        ["median", "mean"],
        loc="upper left",
        fontsize="large",
        framealpha=0.6,
    )

    if y_scale_override:
        ax.set_yscale(y_scale_override)
        ax.get_yaxis().get_major_formatter().labelOnlyBase = False

    else:
        plt.locator_params(axis="y", nbins=10)

    fig = plt.gcf()

    if show_plot:
        plt.show()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=PLOT_DPI)
        plt.close(fig)


def visualize_bar_plot(values, labels, title, show_plot=True, save_path=None):
    """Represent the passed values in a bar plot, showing it and/or saving it to a file"""

    fig, ax = plt.subplots(figsize=PLOT_FIG_SIZE)
    ax.set_title(title, fontsize=PLOT_TITLE_SIZE)

    if not labels:
        labels = range(len(values))

    plt.bar(x=range(len(values)), height=values, tick_label=labels)

    fig = plt.gcf()

    if show_plot:
        plt.show()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=PLOT_DPI)
        plt.close(fig)
