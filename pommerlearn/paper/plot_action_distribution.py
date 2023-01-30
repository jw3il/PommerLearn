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


runs = find_dirs("runs", [f"sl_250s_2false__short_05argmax_2e_{i}" for i in range(0, 3)])

# skip some colors
for a in range(7):
    plt.bar([], [])

patterns = ["", "/" ,  "o", "\\" , "|" , "-" , "+" , "x", "O", ".", "*" ]
iteration_list = [0, 24, 49]
width = 0.8 / len(iteration_list)
for k, iteration in enumerate(iteration_list):
    offset = width * (1 - len(iteration_list)) / 2 + k * width
    action_probs = load_multi_action_probs(runs, iteration)
    plt.bar(np.arange(6) + offset, action_probs.mean(axis=0), width=width, yerr=action_probs.std(axis=0), capsize=width * 20, label=f"Iteration {iteration + 1}")  # , hatch=patterns[k])

plt.ylabel("Proportion")
plt.xticks(np.arange(6), ["Idle", "Up", "Down", "Left", "Right", "Bomb"])
plt.xlabel("Action")
plt.tight_layout()
plt.legend()
# plt.show()
plt.savefig("action_dist.pdf", bbox_inches='tight')
