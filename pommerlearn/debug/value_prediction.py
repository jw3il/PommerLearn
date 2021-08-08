import sys
from pathlib import Path
import numpy as np
import zarr
import matplotlib.pyplot as plt

from torch.distributions.categorical import Categorical
from tqdm import tqdm

from dataset_util import PommerDataset, get_agent_episode_slice, get_agent_died_in_step
import torch

from env.replay_env import PommeReplay
from nn.PommerModel import PommerModel


def plot_value_prediction(dataset_path, model_path, agent_episode, discount_factor, value_version, figure_prefix,
                          save_figure, show_figure):
    z = zarr.open(str(dataset_path), 'r')
    print("Total number of episodes in dataset", len(z.attrs.get('AgentSteps')))

    data = PommerDataset.from_zarr(z, value_version, discount_factor)
    episode_slice = get_agent_episode_slice(z, agent_episode)
    num_steps = episode_slice.stop - episode_slice.start

    print(f"Episode {agent_episode}: Steps {num_steps} from {episode_slice.start} to {episode_slice.stop}, "
          f"val[0] {data[episode_slice.start].val.item()}")

    predicted_values = np.empty(num_steps)

    model = torch.jit.load(f"{model_path}/torch_{'cuda' if torch.cuda.is_available() else 'cpu'}/model-bsize-1.pt")
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    for i in tqdm(range(0, num_steps)):
        obs_tensor = torch.as_tensor(data[episode_slice.start + i].obs)
        # unsqueeze to add batch dimension
        value_out, policy_out = model(PommerModel.flatten(obs_tensor.unsqueeze(0).to(device=device_name), None))
        predicted_values[i] = value_out
        # print(f"Step {i - episode_slice.start}: {value_out, policy_out}")

    steps = np.arange(0, num_steps)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Value')

    real_value = data[episode_slice].val
    ax1.plot(steps, predicted_values, label="predicted values")
    ax1.plot(steps, data[episode_slice].val, label="target values")

    minv = min(predicted_values.min(), real_value.min())
    maxv = max(predicted_values.max(), real_value.max())

    died_in_step = get_agent_died_in_step(
        z.attrs.get('EpisodeActions')[agent_episode], z.attrs.get('EpisodeDead')[agent_episode]
    )
    # aggregate agents according to the steps they died in
    died_in_step_aggregated = {}
    for id, step in enumerate(died_in_step):
        if step != 0:
            if not step in died_in_step_aggregated:
                died_in_step_aggregated[step] = []

            died_in_step_aggregated[step].append(id)

    for step in died_in_step_aggregated:
        agent_ids = died_in_step_aggregated[step]
        plt.vlines(step, minv, maxv, colors="black")
        text = f"{' and '.join([str(id) for id in agent_ids])} died "
        plt.annotate(text, (step, (maxv + minv) / 2), ha='right', fontsize=14)

    fig.tight_layout()
    plt.legend(loc='lower left')

    if save_figure:
        fig.savefig(f"{figure_prefix}_ep{agent_episode}_g{discount_factor:.2f}_v{value_version}.svg", bbox_inches='tight')

    if show_figure:
        plt.show()


def print_episodes_with_value(dataset_path, value_version, target_value):
    episodes = []

    z = zarr.open(str(dataset_path), 'r')
    data = PommerDataset.from_zarr(z, value_version, 1)
    for agent_episode in range(0, len(z.attrs.get('AgentSteps'))):
        episode_slice = get_agent_episode_slice(z, agent_episode)
        if np.abs(data[episode_slice.start].val - target_value) < 0.01:
            episodes.append(agent_episode)

    print(episodes)
    return episodes

# nice test episodes in mcts_data_500:
# 484: loss (second place)
# 530: win stable
# 531: win unstable
# 532: tie between 0 and 1

dataset_path="mcts_data_500.zr"

# print_episodes_with_value(dataset_path, value_version=2, target_value=-1/3)
# PommeReplay.play(zarr.open(str(dataset_path), 'r'), 484, render=True, render_pause=None)
# sys.exit()

plot_value_prediction(
    dataset_path=dataset_path,
    model_path="model__v2_g0.97",
    agent_episode=484,
    discount_factor=0.97,
    value_version=2,
    figure_prefix="values_g0.97",
    save_figure=True,
    show_figure=True
)
