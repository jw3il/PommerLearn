from pathlib import Path
from typing import List, Union, Tuple, Optional

import numpy as np
import torch
from pommerman import constants
import zarr
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


class Samples:
    """
    A container for all samples which belong to a data set.
    """

    def __init__(self, obs: np.ndarray, val: np.ndarray, act: np.ndarray, pol: np.ndarray):
        assert len(obs) == len(val) == len(act) == len(pol), \
            f"Sample array lengths are not the same! Got: {len(obs)}, {len(val)}, {len(act)}, {len(pol)}"

        self.obs = obs
        self.val = val
        self.act = act
        self.pol = pol

    @staticmethod
    def from_zarr(path: Path, discount_factor: float, verbose: bool = False):
        z = zarr.open(str(path), 'r')

        if verbose:
            print(
                f"Opening dataset {str(path)} with {z.attrs['Steps']} samples "
                f"from {len(z.attrs['EpisodeSteps'])} episodes"
            )

        value_target = get_value_target(z, discount_factor)

        z_steps = z.attrs['Steps']
        obs = z['obs'][:z_steps]
        act = z['act'][:z_steps]
        pol = z['pol'][:z_steps]

        return Samples(obs, value_target, act, pol)

    @staticmethod
    def create_empty(count):
        """
        Create an empty sample container.

        :param count: The number of samples
        """
        obs = np.empty((count, 18, 11, 11), dtype=float)
        val = np.empty(count, dtype=float)
        act = np.empty(count, dtype=int)
        pol = np.empty((count, 6), dtype=float)

        return Samples(obs, val, act, pol)

    def split_based_on_idx(self, split_idx: int, from_idx: int = 0, to_idx: int = None):
        """
        Splits samples[from_idx:to_idx] into two halves at the given split index and returns the first and second half

        :param samples: Samples to be split
        :param split_idx: Index where to split
        :param from_idx: The index where the splitting starts
        :param to_idx: The index where the splitting ends (None = until end)
        :return: 1st half, 2nd half after the split
        """
        if to_idx is None:
            to_idx = len(self) - from_idx

        obs_1, obs_2 = split_arr_based_on_idx(self.obs[from_idx:to_idx], split_idx)
        val_1, val_2 = split_arr_based_on_idx(self.val[from_idx:to_idx], split_idx)
        act_1, act_2 = split_arr_based_on_idx(self.act[from_idx:to_idx], split_idx)
        pol_1, pol_2 = split_arr_based_on_idx(self.pol[from_idx:to_idx], split_idx)

        first = Samples(obs_1, val_1, act_1, pol_1)
        second = Samples(obs_2, val_2, act_2, pol_2)
        return first, second

    def to_tensor_dataset(self):
        obs_prime = torch.Tensor(self.obs)
        val_prime = torch.Tensor(self.val)
        act_prime = torch.Tensor(self.act).long()
        pol_prime = torch.Tensor(self.pol)

        return TensorDataset(obs_prime, val_prime, act_prime, pol_prime)

    def set(self, other_samples, to_index, from_index=0, count: Optional[int] = None):
        """
        Sets own_samples[to_index:to_index + count] = other_samples[from_index:from_index + count].

        :param other_samples: The other samples which should overwrite samples in this sample object
        :param to_index: The destination index
        :param from_index: The source index
        :param count: The number of elements which will be copied. If None, all samples will be copied.
        """

        count = len(other_samples) if count is None else count
        self.obs[to_index:to_index + count] = other_samples.obs[from_index:from_index + count]
        self.val[to_index:to_index + count] = other_samples.val[from_index:from_index + count]
        self.act[to_index:to_index + count] = other_samples.act[from_index:from_index + count]
        self.pol[to_index:to_index + count] = other_samples.pol[from_index:from_index + count]

    def append(self, other_samples):
        """
        Appends samples to this sample object.

        :param other_samples: The samples which should be appended to this sample object
        """

        self.obs = np.append(self.obs, other_samples.obs, axis=0)
        self.val = np.append(self.val, other_samples.val, axis=0)
        self.act = np.append(self.act, other_samples.act, axis=0)
        self.pol = np.append(self.pol, other_samples.pol, axis=0)

    def shuffle(self):
        random_state = np.random.get_state()

        def deterministic_shuffle(arr):
            np.random.set_state(random_state)
            np.random.shuffle(arr)

        deterministic_shuffle(self.obs)
        deterministic_shuffle(self.val)
        deterministic_shuffle(self.act)
        deterministic_shuffle(self.val)

    def __len__(self):
        return len(self.obs)


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


def split_arr_based_on_idx(array: np.ndarray, split_idx: int) -> (np.ndarray, np.ndarray):
    """
    Splits a given array into two halves at the given split index and returns the first and second half
    :param array: Array to be split
    :param split_idx: Index where to split
    :return: 1st half, 2nd half after the split
    """
    return array[:split_idx], array[split_idx:]


def get_last_dataset_path(path_infos: List[Union[str, Tuple[str, float]]]) -> str:
    """
    Returns the last path provided in the path_infos list.

    :param path_infos: The path information of the zarr datasets which should be used. Expects a single path or a list
                  containing strings (paths) or a tuple of the form (path, proportion) where 0 <= proportion <= 1
                  is the number of samples which will be selected randomly from this data set.
    :return: The last path
    """
    if isinstance(path_infos, str):
        return path_infos

    last_info = path_infos[len(path_infos) - 1]

    if isinstance(last_info, tuple):
        path, proportion = last_info
        return path
    else:
        return last_info


def create_data_loaders(path_infos: Union[str, List[Union[str, Tuple[str, float]]]], discount_factor: float,
                        test_size: float, batch_size: int, verbose: bool = True) -> [DataLoader, DataLoader]:
    """
    Returns pytorch dataset loaders for a given path

    :param path_infos: The path information of the zarr datasets which should be used. Expects a single path or a list
                      containing strings (paths) or a tuple of the form (path, proportion) where 0 <= proportion <= 1
                      is the number of samples which will be selected randomly from this data set.
    :param discount_factor: The discount factor which should be used
    :param test_size: Percentage of data to use for testing
    :param batch_size: Batch size to use for training
    :param verbose: Log debug information
    :return: Training loader, validation loader
    """

    assert 0 <= test_size < 1, f"Incorrect test size: {test_size}"

    if isinstance(path_infos, str):
        path_infos = [path_infos]

    def get_elems(info):
        if isinstance(info, tuple):
            return info
        else:
            return info, 1

    def get_total_sample_count():
        all_train_samples = 0
        all_test_samples = 0

        for info in path_infos:
            path, proportion = get_elems(info)
            z = zarr.open(str(path), 'r')
            num_samples = int(z.attrs['Steps'] * proportion)

            test_samples = int(num_samples * test_size)
            train_samples = num_samples - test_samples

            all_train_samples += train_samples
            all_test_samples += test_samples

        return all_train_samples, all_test_samples

    total_train_samples, total_test_samples = get_total_sample_count()

    if verbose:
        print(f"Loading {total_train_samples + total_test_samples} samples from {len(path_infos)} dataset(s) with "
              f"test size {test_size}")

    buffer_train = Samples.create_empty(total_train_samples)
    buffer_test = Samples.create_empty(total_test_samples)

    # create a container for all samples
    if verbose:
        print(f"Created buffers with (train: {len(buffer_train)}, test: {len(buffer_test)}) samples")

    buffer_train_idx = 0
    buffer_test_idx = 0
    for info in path_infos:
        path, proportion = get_elems(info)
        elem_samples = Samples.from_zarr(path, discount_factor, verbose)

        assert 0 <= proportion <= 1, f"Invalid proportion {proportion}"

        if proportion < 1:
            elem_samples_nb = int(len(elem_samples) * proportion)
            # TODO: Replace with uniform random sampling. Problem: Shuffling destroys train/test split
            elem_samples_from = np.random.randint(0, len(elem_samples) - elem_samples_nb)

            if verbose:
                print(f"Selected slice [{elem_samples_from}:{elem_samples_from + elem_samples_nb}] "
                      f"({elem_samples_nb} samples)")
        else:
            elem_samples_nb = len(elem_samples)
            elem_samples_from = 0

        test_nb = int(elem_samples_nb * test_size)
        train_nb = elem_samples_nb - test_nb

        buffer_train.set(elem_samples, buffer_train_idx, elem_samples_from, train_nb)
        buffer_test.set(elem_samples, buffer_test_idx, elem_samples_from + train_nb, test_nb)

        # copy first num_samples samples
        if verbose:
            print("Copied {} ({}, {}) samples ({:.2f}%) to buffers @ ({}, {})"
                  .format(elem_samples_nb, train_nb, test_nb, proportion * 100, buffer_train_idx, buffer_test_idx))

        buffer_train_idx += train_nb
        buffer_test_idx += test_nb

    assert buffer_train_idx == total_train_samples and buffer_test_idx == total_test_samples, \
        f"The number of copied samples is wrong.. " \
        f"{(buffer_train_idx, total_train_samples, buffer_test_idx, total_test_samples)}"

    def get_loader(samples: Samples):
        return DataLoader(samples.to_tensor_dataset(), batch_size=batch_size, shuffle=True)

    if verbose:
        print("Creating DataLoaders..", end='')

    train_loader = get_loader(buffer_train)
    test_loader = get_loader(buffer_test) if total_test_samples > 0 else None

    if verbose:
        print(" done.")

    return train_loader, test_loader


def log_dataset_stats(path, log_dir, iteration):
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
