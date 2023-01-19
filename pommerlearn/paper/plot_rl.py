import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, ScalarEvent

from paper.util import get_label

from matplotlib_settings import set_matplotlib_font_size

set_matplotlib_font_size(12, 14, 16)

def find_dirs(base_dir, endswith_list):
    endswith_list = endswith_list.copy()
    found_paths = []
    for file in os.listdir(base_dir):
        for i in reversed(range(0, len(endswith_list))):
            endswith = endswith_list[i]
            if file.endswith(endswith):
                found_paths.append(os.path.join(base_dir, file))
                del endswith_list[i]

    if len(endswith_list) > 0:
        print(f"Warning: did not find {endswith_list}.")

    return found_paths


def scalars_to_xy(tb_scalar_event_list: List[ScalarEvent]):
    x = np.zeros(len(tb_scalar_event_list))
    y = np.zeros(len(tb_scalar_event_list))
    for i, e in enumerate(tb_scalar_event_list):
        x[i] = e.step
        y[i] = e.value
    return (x, y)


def load_win_rate(path):
    acc = EventAccumulator(path)
    acc.Reload()
    return scalars_to_xy(acc.Scalars("Dataset/Win ratio 0"))


def aggregate(xy_list, label):
    # first filter out runs with invalid length
    max_x_len = max(len(x) for (x, y) in xy_list)
    for i in reversed(range(0, len(xy_list))):
        if len(xy_list[i][0]) != max_x_len:
            print(f"Warning: length of element {i} is {len(xy_list[i][0])} != {max_x_len}. Will be ignored. Label: {label}")
            del xy_list[i]

    common_x = xy_list[0][0]
    assert np.array([(x == common_x) for (x, y) in xy_list]).all(), "X does not match"

    y_array = np.empty((len(xy_list), len(xy_list[0][0])))
    for i, (x, y) in enumerate(xy_list):
        y_array[i] = y

    plt.fill_between(common_x, y_array.mean(axis=0) - y_array.std(axis=0), y_array.mean(axis=0) + y_array.std(axis=0), alpha=0.3)
    plt.plot(common_x, y_array.mean(axis=0), label=label)


runs = find_dirs("runs", [f"dummy_250s_2false_short_noterm_noDiscount_{i}" for i in range(1, 6)])
aggregate([load_win_rate(p) for p in runs], label=get_label("OnePlayer", "dummy", None))

runs = find_dirs("runs", [f"sl_250s_2false_short_noterm_noDiscount_{i}" for i in range(1, 6)])
aggregate([load_win_rate(p) for p in runs], label=get_label("OnePlayer", "sl", None))

runs = find_dirs("runs", [f"dummy_250s_2true_short_noterm_noDiscount_{i}" for i in range(1, 6)])
aggregate([load_win_rate(p) for p in runs], label=get_label("TwoPlayer", "dummy", None))

runs = find_dirs("runs", [f"sl_250s_2true_short_noterm_noDiscount_{i}" for i in range(1, 6)])
aggregate([load_win_rate(p) for p in runs], label=get_label("TwoPlayer", "sl", None))

# plt.ylim(0, 1.0)
plt.xlabel("Iterations")
plt.ylabel("Win rate against $\\texttt{Simple}_\\texttt{C}$ opponents", labelpad=8)
plt.legend()
plt.savefig("ffa_rl.pdf", bbox_inches="tight")
plt.show()
