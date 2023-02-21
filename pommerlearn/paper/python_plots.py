import numpy as np
import pandas as pd
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path


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


def plotting_data(exp_path: Path):
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
    heatmap_rotated = np.zeros((4,11,11))
    for game_positons in positions_experiment:
        for agent_ix,positions in enumerate(game_positons):
            game_heatmap = np.zeros((11,11))
            for pos in positions:
                game_heatmap[pos[0], pos[1]] += 1
            
            heatmap[agent_ix] += game_heatmap
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
    
    return heatmap, state_varieties, heatmap_rotated


def plot(experiments, agents):
    """plot data against simple agent for SP runs"""
    heats = np.zeros((len(agents),4,11,11))
    heats_rotated = np.zeros((len(agents),4,11,11))
    varieties = np.zeros((len(agents),4,20))
    for i,pre in enumerate(agents):
        # repeat for 2 different experiments
        for experiment in experiments:
            # aggregate heatmaps and state varieties over all games
            if f'{pre}' != experiment.name[:len(pre)] or 'raw' in experiment.name or '2p' in experiment.name:
                # Skip RawNet and MP experiments 
                continue
            print('Getting info from ', experiment)
            hmap, game_varieties, hmap_rot = plotting_data(experiment)
            heats[i] += hmap
            varieties[i] += game_varieties
            heats_rotated[i] += hmap_rot

    agents = [agent[:-1] for agent in agents]

    def heatmap(heat, name, norm=None, cmap=None):
        fig, ax = plt.subplots()
        im = ax.imshow(heat, cmap=cmap, norm=norm)
        fig.colorbar(im)
        plt.savefig(f"{name}.png")
        plt.clf()
    
    for i,pre in enumerate(agents):
        hm = heats[i][0]
        heatmap(hm/np.sum(hm)*100, f"{pre}")
    
    for i,pre in enumerate(agents):
        hm = heats_rotated[i][0]
        heatmap(hm/np.sum(hm)*100, f"{pre}_rotated")

    for i in range(len(agents)):
        plt.bar(range(1,21), varieties[i][0]/varieties[i][0].sum(), alpha=0.6)
    plt.legend(agents)
    plt.savefig(f"state_variety.png")
    plt.clf()

    for i in range(len(agents)):
        for j in range(i+1, len(agents)):
            hm = heats[i][0]/np.sum(heats[i][0]) - heats[j][0]/np.sum(heats[j][0])
            heatmap(hm, f"{agents[i]}_vs_{agents[j]}", MidpointNormalize(vmin=hm.min(), vmax=hm.max(), midpoint=0), 'RdBu_r')


if __name__ == "__main__":
    eval_path = Path('./py-eval-nopickle')
    experiments = [x for x in eval_path.iterdir() if x.is_dir()]

    agents = ['sl2rl-', 'rl-', 'sl-'] # str to filter experiment folders
    plot(experiments, agents) # list with 2 agents assumes foldernames to be 
