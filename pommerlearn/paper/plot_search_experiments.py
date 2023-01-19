import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper.util import get_label

from matplotlib_settings import set_matplotlib_font_size

set_matplotlib_font_size(14, 16, 18)

df = pd.read_csv("20221231_103020_pommer_log.csv")

for i, search_mode in enumerate(["OnePlayer", "TwoPlayer"]):
    for j, model_name in enumerate(["dummy", "sl"]):
        color = plt.cm.tab10.colors[i * 2 + j]
        for k, terminal_solver in enumerate([True, False]):
            filtered = df[(df.SearchMode == search_mode) & (df.ModelName == model_name) & (df.TerminalSolver == terminal_solver)]
            if k == 0:
                style = "-"
            else:
                style = "--"

            plt.plot(filtered.Simulations, filtered.Wins0 / filtered.Episodes, color=color, linestyle=style, label=get_label(search_mode, model_name, terminal_solver), marker='o')
            sim_ticks = filtered.Simulations

plt.xlabel("Simulations per step")
plt.ylabel("Win rate against $\\texttt{Simple}_\\texttt{C}$ opponents", labelpad=8)
plt.xticks(sim_ticks, sim_ticks)
y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
plt.yticks(y_ticks, y_ticks)
plt.legend(bbox_to_anchor=(1.0, 1.02))

# plt.show()
plt.savefig("search_experiments.pdf", bbox_inches='tight')
