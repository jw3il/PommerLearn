import numpy as np
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, HistogramEvent, \
    STORE_EVERYTHING_SIZE_GUIDANCE

from paper.matplotlib_settings import init_plt, set_matplotlib_font_size
from paper.util import find_dirs

init_plt()
set_matplotlib_font_size(14, 16, 18)
plt.rcParams['figure.figsize'] = [6.5, 5]


def get_action_counts(histogram_event: HistogramEvent):
    value = histogram_event.histogram_value
    action_counts = np.zeros(6)

    last_limit = -float('inf')
    for (limit, count) in zip(value.bucket_limit, value.bucket):
        if count > 0:
            index = int(np.floor(limit))
            print(f"{index} ({last_limit:.2f}-{limit:.2f}): {count}")
            action_counts[index] += count

        last_limit = limit

    print(action_counts)
    return action_counts


def load_action_probs(path, iteration):
    acc = EventAccumulator(path, size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE)
    acc.Reload()
    action_counts = get_action_counts(acc.Histograms("Dataset/Actions 0")[iteration])
    return action_counts / action_counts.sum()


def load_multi_action_probs(paths, iteration):
    action_probs = np.zeros((len(paths), 6))
    for i, path in enumerate(paths):
        action_probs[i] = load_action_probs(path, iteration)
    return action_probs


def generate_plot(dir_list, filename="action_dist.pdf", iterations=(0, 25, 50), show_legend=True):
    # skip some colors
    for a in range(7):
        plt.bar([], [])

    patterns = ["", "/" ,  "o", "\\" , "|" , "-" , "+" , "x", "O", ".", "*" ]
    width = 0.8 / len(iterations)
    for k, iteration in enumerate(iterations):
        offset = width * (1 - len(iterations)) / 2 + k * width
        action_probs = load_multi_action_probs(dir_list, iteration)
        plt.bar(np.arange(6) + offset, action_probs.mean(axis=0), width=width, yerr=action_probs.std(axis=0), capsize=width * 20, label=f"Iteration {iteration}")  # , hatch=patterns[k])

    plt.ylabel("Proportion")
    plt.xticks(np.arange(6), ["Idle", "Up", "Down", "Left", "Right", "Bomb"])
    plt.xlabel("Action")
    plt.tight_layout()
    if show_legend:
        plt.legend()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


generate_plot(
    find_dirs("runs", [f"sl_250s_2false_2ep_{i}" for i in range(0, 5)]),
    "action_dist_rl_regular.pdf"
)

generate_plot(
    find_dirs("runs", [f"sl_250s_2false_2ep_05argmax_{i}" for i in range(0, 5)]),
    "action_dist_rl_exploit.pdf"
)

generate_plot(
    find_dirs("runs", [f"sl_250s_2false_2ep_{i}" for i in range(0, 5)]),
    "action_dist_rl_regular_first10.pdf",
    iterations=list(range(11)),
    show_legend=False
)
