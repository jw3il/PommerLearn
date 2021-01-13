from pathlib import Path

import numpy as np
import torch
from pommerman import constants
import zarr
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


def get_agent_actions(z, episode):
    episode_steps = z.attrs['EpisodeSteps'][episode]
    actions = np.ones((episode_steps, 4)) * constants.Action.Stop.value
    raw_actions = z.attrs['EpisodeActions'][episode]

    for a in range(0, 4):
        actions[0:len(raw_actions[a]), a] = raw_actions[a]

    return actions


def get_agent_episode_slice(z, agent_episode):
    # sum up all steps up to our episode
    start_index = int(np.sum(z.attrs['AgentSteps'][0:agent_episode]))
    # add the amount of steps of the episode
    end_index = start_index + z.attrs['AgentSteps'][agent_episode]
    return slice(start_index, end_index)


def last_episode_is_cut(z):
    return np.sum(z.attrs.get('AgentSteps')) != z.attrs.get('Steps')


def get_value_target(z, discount_factor: float) -> np.ndarray:
    """
    Creates the value target for a zarr dataset z.

    :param z: The zarr dataset
    :param discount_factor: The discount factor
    :return: The value target for z
    """
    total_steps = z.attrs.get('Steps')
    agent_steps = np.array(z.attrs.get('AgentSteps'))
    agent_ids = np.array(z.attrs.get('AgentIds'))
    agent_episode = np.array(z.attrs.get('AgentEpisode'))
    episode_winner = np.array(z.attrs.get('EpisodeWinner'))
    episode_dead = np.array(z.attrs.get('EpisodeDead'))
    # episode_draw = np.array(z.attrs.get('EpisodeDraw'))
    # episode_done = np.array(z.attrs.get('EpisodeDone'))

    val_target = np.empty(total_steps)
    current_step = 0
    for agent_ep_idx in range(0, len(agent_steps)):
        agent_id = agent_ids[agent_ep_idx]
        ep = agent_episode[agent_ep_idx]
        steps = agent_steps[agent_ep_idx]
        winner = episode_winner[ep]
        dead = episode_dead[ep][agent_id]

        # TODO: Adapt for team mode
        if winner == agent_id:
            episode_reward = 1
        elif dead:
            episode_reward = -1
        else:
            # episode not done
            episode_reward = 0

        # min to handle cut datasets
        next_step = min(current_step + steps, total_steps)
        num_steps = next_step - current_step

        # calculate discount factors backwards
        discounting = np.power(discount_factor, np.arange(steps - 1, steps - 1 - num_steps, -1))
        val_target[current_step:next_step] = discounting * episode_reward

        current_step = next_step

    return val_target


def split_based_on_idx(array: np.ndarray, split_idx: int) -> (np.ndarray, np.ndarray):
    """
    Splits a given array into two halves at the given split index and returns the first and second half
    :param array: Array to be split
    :param split_idx: Index where to split
    :return: 1st half, 2nd half after the split
    """
    return array[:split_idx], array[split_idx:]


def create_data_loaders(path: Path, discount_factor: float, test_size: float, batch_size: int) -> [DataLoader, DataLoader]:
    """
    Returns pytorch dataset loaders for a given path

    :param path: The path of a zarr dataset
    :param discount_factor: The discount factor which should be used
    :param test_size: Percentage of data to use for testing
    :param batch_size: Batch size to use for training
    :return: Training loader, Validation loader
    """
    z = zarr.open(str(path), 'r')

    print(f"Opening dataset {str(path)} with {z.attrs['Steps']} samples from {len(z.attrs['EpisodeSteps'])} episodes")

    value_target = get_value_target(z, discount_factor)

    z_steps = z.attrs['Steps']
    obs = z['obs'][:z_steps]
    act = z['act'][:z_steps]
    pol = z['pol'][:z_steps]
    if len(value_target) != z_steps:
        raise ValueError(f"Value target size does not match dataset size! Got {len(value_target)}, expected {z_steps}.")

    def get_loader(x, yv, ya, yp):
        x = torch.Tensor(x)
        yv = torch.Tensor(yv)
        ya = torch.Tensor(ya).long()
        yp = torch.Tensor(yp)
        return DataLoader(TensorDataset(x, yv, ya, yp), batch_size=batch_size, shuffle=True)

    if test_size == 0:
        return get_loader(obs, value_target, act, pol), None

    if 0 < test_size < 1:
        split_idx = len(obs) - int(test_size * len(obs))
        x_train, x_val = split_based_on_idx(obs, split_idx)
        yv_train, yv_val = split_based_on_idx(value_target, split_idx)
        ya_train, ya_val = split_based_on_idx(act, split_idx)
        yp_train, yp_val = split_based_on_idx(pol, split_idx)
        return get_loader(x_train, yv_train, ya_train, yp_train), get_loader(x_val, yv_val, ya_val, yp_val)

    raise ValueError(f"Incorrect test size: {test_size}")


def log_dataset_stats(path: Path, log_dir, iteration):
    """
    Log dataset stats to tensorboard.

    :param path: The path of a zarr dataset
    :param log_dir: The logdir of the summary writer
    :param iteration: The iteration this dataset belongs to
    """
    z = zarr.open(str(path), 'r')

    writer = SummaryWriter(log_dir=log_dir)

    z_steps = z.attrs['Steps']
    steps = np.array(z.attrs["EpisodeSteps"])
    winners = np.array(z.attrs["EpisodeWinner"])
    done = np.array(z.attrs["EpisodeDone"])
    actions = z.attrs["EpisodeActions"]

    pol = z['pol'][:z_steps]
    pol_entropy = -np.sum(pol * np.log(pol, out=np.zeros_like(pol), where=(pol != 0)), axis=1)
    writer.add_scalar("Dataset/Policy entropy mean", pol_entropy.mean(), iteration)
    writer.add_scalar("Dataset/Policy entropy std", pol_entropy.std(), iteration)
    writer.add_text("Dataset/Policy NaN", str(np.isnan(pol).any()), iteration)

    num_episodes = len(steps)

    writer.add_scalar("Dataset/Episodes", num_episodes, iteration)
    writer.add_scalar("Dataset/Steps mean", steps.mean(), iteration)
    writer.add_scalar("Dataset/Steps std", steps.std(), iteration)

    for a in range(0, 4):
        winner_a = np.sum(winners[:] == a)
        writer.add_scalar(f"Dataset/Win ratio {a}", winner_a / num_episodes, iteration)

        actions_a = []
        episode_steps = np.empty(len(actions))
        for i, ep in enumerate(actions):
            ep_actions = ep[a]
            episode_steps[i] = len(ep_actions)
            actions_a += ep_actions

        writer.add_scalar(f"Dataset/Steps mean {a}", episode_steps.mean(), iteration)

        # TODO: Correct bin borders
        writer.add_histogram(f"Dataset/Actions {a}", np.array(actions_a), iteration)

    no_winner = np.sum((winners == -1) * (done == True))
    writer.add_scalar(f"Dataset/Draw ratio", no_winner / num_episodes, iteration)

    not_done = np.sum(done == False)
    writer.add_scalar(f"Dataset/Not done ratio", not_done / num_episodes, iteration)

    writer.close()
