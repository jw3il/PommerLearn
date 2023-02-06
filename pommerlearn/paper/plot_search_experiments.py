import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper.util import get_label

from matplotlib_settings import set_matplotlib_font_size, init_plt

init_plt()
set_matplotlib_font_size(14, 16, 18)
plt.rcParams['figure.figsize'] = [6.5, 5]

df = pd.read_csv("20230205_004140_pommer_log.csv")
df["Supervised"] = df.ModelName.str.startswith("sl")
df["RelativeWin0"] = df.Wins0 / df.Episodes
df_agg = df.groupby(["SearchMode", "Supervised", "Simulations"]).agg({'RelativeWin0': ['mean', 'std']}).reset_index()

for i, search_mode in enumerate(["OnePlayer", "TwoPlayer"]):
    for j, supervised in enumerate([False, True]):
        color = plt.cm.tab10.colors[i * 2 + j]
        filtered = df_agg[(df_agg.SearchMode == search_mode) & (df_agg.Supervised == supervised)]
        model_name = "sl" if supervised else ""
        plt.fill_between(filtered.Simulations, filtered.RelativeWin0["mean"] - filtered.RelativeWin0["std"],  filtered.RelativeWin0["mean"] + filtered.RelativeWin0["std"], color=color, alpha=0.3)
        plt.plot(filtered.Simulations, filtered.RelativeWin0["mean"], color=color, label=get_label(search_mode, model_name, False), marker='o')
        sim_ticks = filtered.Simulations

plt.xlabel("Simulations per step")
plt.ylabel("Win rate against $\\texttt{Simple}_\\texttt{C}$ opponents", labelpad=8)
plt.xticks(sim_ticks, sim_ticks)
y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
plt.yticks(y_ticks, y_ticks)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2)

plt.tight_layout()
plt.savefig("search_experiments.pdf", bbox_inches='tight')
plt.show()
