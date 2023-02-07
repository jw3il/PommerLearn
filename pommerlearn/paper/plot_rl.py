import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, ScalarEvent

from paper.util import get_label, find_dirs

from matplotlib_settings import set_matplotlib_font_size, init_plt

init_plt()
set_matplotlib_font_size(14, 16, 18)
plt.rcParams['figure.figsize'] = [6.5, 5]


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


runs = find_dirs("runs", [f"dummy_250s_2false__short_05argmax_2e_{i}" for i in range(0, 3)])
aggregate([load_win_rate(p) for p in runs], label=get_label("OnePlayer", "dummy", None))

runs = find_dirs("runs", [f"sl_250s_2false__short_05argmax_2e_{i}" for i in range(0, 3)])
aggregate([load_win_rate(p) for p in runs], label=get_label("OnePlayer", "sl", None))

runs = find_dirs("runs", [f"dummy_250s_2true__short_05argmax_2e_{i}" for i in range(0, 3)])
aggregate([load_win_rate(p) for p in runs], label=get_label("TwoPlayer", "dummy", None))

runs = find_dirs("runs", [f"sl_250s_2true__short_05argmax_2e_{i}" for i in range(0, 3)])
aggregate([load_win_rate(p) for p in runs], label=get_label("TwoPlayer", "sl", None))

plt.ylim(0, 1.0)
plt.xlabel("Training iterations")
plt.ylabel("Win rate against $\\texttt{Simple}_\\texttt{C}$ opponents", labelpad=8)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2)
plt.tight_layout()
plt.savefig("ffa_rl.pdf", bbox_inches="tight")
plt.show()
