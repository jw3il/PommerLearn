import numpy as np
import pandas as pd
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

from paper.matplotlib_settings import init_plt, set_matplotlib_font_size
from paper.util import get_label

init_plt()
set_matplotlib_font_size(16, 18, 20)
plt.rcParams['figure.figsize'] = [6.5, 5]

class MidpointNormalize(mpl.colors.Normalize):
    # from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))


def get_plotting_data(exp_path: Path):
    json_path = exp_path / f'{exp_path.name}_json'
    game_files = [x/'game_state.json' for x in json_path.iterdir() if x.is_dir()]

    def game_positions(game):
        # agent positions: agent_id x list(positions)
        positions = [[],[],[],[]]
        for state in sorted(game['state'], key=lambda x: int(x["step_count"])):
            agents = json.loads(state['agents'])
            for agent in agents:
                if agent["is_alive"]:
                    positions[agent["agent_id"]].append(tuple(agent["position"]))
        return positions
    
    positions_experiment = [] #[game][agent_id]->list(positions)
    for gf in game_files:
        with gf.open() as f:
            game = json.load(f)
        positions_game = game_positions(game)
        positions_experiment.append(positions_game)

    rotation = {
        (1,1): 0,
        (9,1): 3,
        (1,9): 1,
        (9,9): 2,
    }
    # collect heatmap data
    heatmap = np.zeros((4,11,11))
    heatmap_episode = np.zeros((4,11,11))
    heatmap_rotated = np.zeros((4,11,11))
    for game_positons in positions_experiment:
        for agent_ix,positions in enumerate(game_positons):
            game_heatmap = np.zeros((11,11))
            game_heatmap_episode = np.zeros((11, 11))
            for pos in positions:
                game_heatmap[pos[0], pos[1]] += 1
                game_heatmap_episode[pos[0], pos[1]] = 1
            
            heatmap[agent_ix] += game_heatmap
            heatmap_episode[agent_ix] += game_heatmap_episode
            heatmap_rotated[agent_ix] += np.rot90(game_heatmap, k=rotation[positions[0]])
    
    # collect state varieties for one game
    max_variety = 20
    state_varieties = np.zeros((4,max_variety))
    for game_positons in positions_experiment:
        for agent_ix, positions in enumerate(game_positons):
            # add seen varieties
            for i in range(max_variety, len(positions)):
                window = positions[i-max_variety: i]
                variety = len(set(window))
                state_varieties[agent_ix, variety-1] += 1
    
    return heatmap, state_varieties, heatmap_rotated, heatmap_episode


def get_aggregated_data(experiments, agent_prefix):
    """plot data against simple agent for SP runs"""
    heats = np.zeros((len(agent_prefix),4,11,11))
    heats_eps = np.zeros((len(agent_prefix),4,11,11))
    heats_rotated = np.zeros((len(agent_prefix),4,11,11))
    varieties = np.zeros((len(agent_prefix),4,20))
    for i, pre in enumerate(agent_prefix):
        # repeat for 2 different experiments
        for experiment in experiments:
            # aggregate heatmaps and state varieties over all games
            if f'{pre}' != experiment.name[:len(pre)] or 'raw' in experiment.name or '2p' in experiment.name:
                # Skip RawNet and MP experiments
                continue
            print('Getting info from ', experiment)
            hmap, game_varieties, hmap_rot, hmap_eps = get_plotting_data(experiment)
            heats[i] += hmap
            heats_eps[i] += hmap_eps
            varieties[i] += game_varieties
            heats_rotated[i] += hmap_rot

    return heats, heats_eps, heats_rotated, varieties


def heatmap(heat, filename, norm=None, cmap=None, label=None, labelpad=22):
    fig, ax = plt.subplots()
    im = ax.imshow(heat, cmap=cmap, norm=norm)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    cbar = fig.colorbar(im, pad=0.02)
    if label is not None:
        cbar.set_label(label, rotation=270, labelpad=labelpad)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.clf()


def plot_debug(experiments, agents, format='png', num_games=500):
    heats, heats_eps, heats_rotated, varieties = get_aggregated_data(experiments, agents)
    agents = [agent[:-1] for agent in agents]

    for i,pre in enumerate(agents):
        hm = heats[i][0]
        heatmap(hm / num_games, f"py_{pre}.{format}", label="Average steps on position per game", cmap="viridis")

    for i, pre in enumerate(agents):
        hm = heats_eps[i][0]
        heatmap(hm / num_games, f"py_{pre}_eps.{format}", label="Average position occupancy per game", cmap="viridis")

    for i,pre in enumerate(agents):
        hm = heats_rotated[i][0]
        heatmap(hm / num_games, f"py_{pre}_rotated.{format}", label="Average steps on position per game", cmap="viridis")

    for i in range(len(agents)):
        plt.bar(range(1,21), varieties[i][0]/varieties[i][0].sum(), alpha=0.6)

    plt.legend(agents)
    plt.tight_layout()
    plt.savefig(f"py_state_variety.{format}", bbox_inches="tight")
    plt.clf()

    for i in range(len(agents)):
        for j in range(i+1, len(agents)):
            hm = heats[i][0]/np.sum(heats[i][0]) - heats[j][0]/np.sum(heats[j][0])
            heatmap(hm, f"py_{agents[i]}_vs_{agents[j]}", MidpointNormalize(vmin=hm.min(), vmax=hm.max(), midpoint=0), 'RdBu_r')


def plot_paper(experiments, num_games=500, format="png"):
    agents = ["sl-", "rl-"]
    heats, heats_eps, heats_rotated, varieties = get_aggregated_data(experiments, agents)
    agents = [agent[:-1] for agent in agents]

    set_matplotlib_font_size(19, 21, 23)

    # individual plots
    for i,pre in enumerate(agents):
        hm = heats_rotated[i][0]
        heatmap(hm / num_games, f"pyp_{pre}_rotated.{format}", label="Average steps on position per game", cmap="viridis", labelpad=26)

    set_matplotlib_font_size(17, 19, 21)

    # state variety comparison
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex='all')
    xticks = range(1,21)
    xtick_labels = ["1"] + [""] * 3 + ["5"] + [""] * 4 + ["10"] + [""] * 4 + ["15"] + [""] * 4 + ["20"]
    axes[0].bar(xticks, varieties[0][0]/varieties[0][0].sum(), color="tab:orange")
    axes[0].xaxis.set_ticks(xticks)
    axes[0].set_ylabel("Proportion")

    axes[1].bar(xticks, varieties[1][0]/varieties[1][0].sum(), color="tab:blue")
    axes[1].xaxis.set_ticks(xticks)
    axes[1].xaxis.set_ticklabels(xtick_labels)
    axes[1].set_xlabel("Unique visited positions within 20 steps")
    axes[1].set_ylabel("Proportion") #, labelpad=17.5)

    fig.legend([get_label("OnePlayer", "(SL)", False), get_label("OnePlayer", "(RL)", False)], loc='upper center', ncol=2, bbox_to_anchor=(0.56, 1.06))
    fig.tight_layout()
    plt.savefig(f"pyp_state_variety.{format}", bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    eval_path = Path('./20230220_py_eval_100')
    experiments = [x for x in eval_path.iterdir() if x.is_dir()]

    agents = ['sl2rl-', 'rl-', 'sl-'] # str to filter experiment folders
    # plot_debug(experiments, agents)
    plot_paper(experiments, format="pdf")
