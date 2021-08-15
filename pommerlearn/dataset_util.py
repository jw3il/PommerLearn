from collections import namedtuple
from pathlib import Path
from typing import List, Union, Tuple, Optional, NamedTuple

import numpy as np
import torch
from pommerman import constants
import zarr
from torch.utils.data import DataLoader, TensorDataset, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter


class PommerSample(NamedTuple):
    """
    Holds a single sample or a batch of samples.
    """
    obs: torch.Tensor
    val: torch.Tensor
    act: torch.Tensor
    pol: torch.Tensor

    @staticmethod
    def merge(a, b):
        """
        Merges two samples and adds the batch dimension if necessary.

        :param a: A sample or batch of samples
        :param b: A sample or batch of samples
        :returns: a batch of samples containing a and b
        """
        if not a.is_batch():
            a = a.expand_batch_dim()
        if not b.is_batch():
            b = b.expand_batch_dim()

        return PommerSample(
            torch.cat((a.obs, b.obs), dim=0),
            torch.cat((a.val, b.val), dim=0),
            torch.cat((a.act, b.act), dim=0),
            torch.cat((a.pol, b.pol), dim=0)
        )

    def is_batch(self):
        return len(self.obs.shape) == 4

    def expand_batch_dim(self):
        """
        Creates a new sample with added batch dimension.

        :returns: this sample expanded by the batch dimension.
        """
        assert not self.is_batch()
        return PommerSample(
            self.obs.unsqueeze(0),
            self.val.unsqueeze(0),
            self.act.unsqueeze(0),
            self.pol.unsqueeze(0)
        )

    def batch_at(self, index):
        """
        Select a single sample from this batch according to the given index.

        :param index: The index of the sample within this batch
        :returns: a new sample with data from the specified index (no batch dimension)
        """
        assert self.is_batch()
        return PommerSample(
            self.obs[index],
            self.val[index],
            self.act[index],
            self.pol[index]
        )

    def equals(self, other):
        return self.obs.shape == other.obs.shape \
               and (self.val == other.val).all() \
               and (self.act == other.act).all() \
               and (self.pol == other.pol).all() \
               and (self.obs == other.obs).all()


class PommerDataset(Dataset):
    """
    A pommerman dataset.
    """
    PLANE_HORIZONTAL_BOMB_MOVEMENT = 7
    PLANE_VERTICAL_BOMB_MOVEMENT = 8
    PLANE_AGENT0 = 10
    PLANE_AGENT1 = 11
    PLANE_AGENT2 = 12
    PLANE_AGENT3 = 13

    def __init__(self, obs, val, act, pol, ids, steps_to_end, transform=None, sequence_length=None, return_ids=False):
        assert len(obs) == len(val) == len(act) == len(pol), \
            f"Sample array lengths are not the same! Got: {len(obs)}, {len(val)}, {len(act)}, {len(pol)}"

        if not isinstance(obs, np.ndarray)\
                or not isinstance(val, np.ndarray)\
                or not isinstance(act, np.ndarray)\
                or not isinstance(pol, np.ndarray):
            assert False, "Invalid data type!"

        self.sequence_length = sequence_length
        self.episode = None

        if self.sequence_length is not None:
            assert self.sequence_length >= 1, "Invalid sequence length!"

        self.obs = obs
        self.val = val
        self.act = act
        self.pol = pol
        self.ids = ids
        self.steps_to_end = steps_to_end

        self.return_ids = return_ids
        self.transform = transform

    @staticmethod
    def from_zarr_path(path: Path, value_version: int, discount_factor: float, mcts_val_weight: Optional[float],
                       transform=None, return_ids=False, verbose: bool = False):
        z = zarr.open(str(path), 'r')
        return PommerDataset.from_zarr(z, value_version, discount_factor, mcts_val_weight, transform, return_ids,
                                       verbose)

    @staticmethod
    def from_zarr(z: zarr.Group, value_version: int, discount_factor: float, mcts_val_weight: Optional[float],
                  transform=None, return_ids=False, verbose: bool = False):
        if verbose:
            print(
                f"Opening dataset {str(z.path)} with {z.attrs['Steps']} samples "
                f"from {len(z.attrs['EpisodeSteps'])} episodes"
            )

        z_steps = z.attrs['Steps']

        return PommerDataset(
            obs=z['obs'][:z_steps],
            val=get_value_target(z, value_version, discount_factor, mcts_val_weight),
            act=z['act'][:z_steps],
            pol=z['pol'][:z_steps],
            ids=get_unique_agent_episode_id(z),
            transform=transform,
            return_ids=return_ids,
            steps_to_end=get_steps_until_end(z)
        )

    @staticmethod
    def create_empty(count, transform=None, sequence_length=None, return_ids=False):
        """
        Create an empty sample container.

        :param count: The number of samples
        """

        return PommerDataset(
            obs=np.empty((count, 18, 11, 11), dtype=float),
            val=np.empty(count, dtype=float),
            act=np.empty(count, dtype=int),
            pol=np.empty((count, 6), dtype=float),
            ids=np.empty(count, dtype=int),
            steps_to_end=np.empty(count, dtype=int),
            transform=transform,
            sequence_length=sequence_length,
            return_ids=return_ids
        )

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
        self.ids[to_index:to_index + count] = other_samples.ids[from_index:from_index + count]
        self.steps_to_end[to_index:to_index + count] = other_samples.steps_to_end[from_index:from_index + count]

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.sequence_length is None:
            # return a single sample

            sample = PommerSample(
                torch.tensor(self.obs[idx], dtype=torch.float),
                torch.tensor(self.val[idx], dtype=torch.float),
                torch.tensor(self.act[idx], dtype=torch.int),
                torch.tensor(self.pol[idx], dtype=torch.float),
            )

            if self.transform is not None:
                sample = self.transform(sample)

            if self.return_ids:
                return (self.ids[idx], *sample)
            else:
                return sample
        else:
            # build a sequence of samples
            sequence = PommerSample(
                torch.zeros((self.sequence_length, 18, 11, 11), dtype=torch.float),
                torch.zeros(self.sequence_length, dtype=torch.float),
                torch.zeros(self.sequence_length, dtype=torch.int),
                torch.zeros((self.sequence_length, 6), dtype=torch.float)
            )

            # check if we have to stop before sequence_length samples
            current_id = self.ids[idx]
            end_idx = idx
            for seq_idx in range(1, self.sequence_length):
                data_idx = idx + seq_idx

                if data_idx >= len(self.obs) or self.ids[data_idx] != current_id:
                    # we reached a different episode / the beginning of the dataset
                    break

                end_idx = data_idx

            # TODO: Use PackedSequence instead of manual padding?
            seq_end = end_idx - idx + 1
            sequence.obs[0:seq_end] = torch.tensor(self.obs[idx:end_idx+1], dtype=torch.float)
            sequence.val[0:seq_end] = torch.tensor(self.val[idx:end_idx+1], dtype=torch.float)
            sequence.act[0:seq_end] = torch.tensor(self.act[idx:end_idx+1], dtype=torch.int)
            sequence.pol[0:seq_end] = torch.tensor(self.pol[idx:end_idx+1], dtype=torch.float)

            if self.transform is not None:
                sequence = self.transform(sequence)

            if self.return_ids:
                ret_ids = torch.ones(self.sequence_length, dtype=torch.int, requires_grad=False) * -1
                ret_ids[0:seq_end] = current_id
                return (ret_ids, *sequence)
            else:
                return sequence


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


def get_unique_agent_episode_id(z) -> np.ndarray:
    """
    Creates unique ids for every new agent episode in the environment and returns an array containing the id of each
    individual step.

    :param z: The zarr dataset
    :return: The unique agent episode ids in z
    """
    total_steps = z.attrs.get('Steps')
    agent_steps = np.array(z.attrs.get('AgentSteps'))

    ids = np.empty(total_steps, dtype=np.int)
    current_id = 0
    current_step = 0

    for steps in agent_steps:
        end = min(current_step + steps, total_steps)
        ids[current_step:end] = current_id
        current_step += steps
        current_id += 1

    return ids


def get_steps_until_end(z) -> np.ndarray:
    """
    Creates an array that contains the number of steps until the last step for each agent episode.

    :param z: The zarr dataset
    :return: The number of steps until the episode of this sample ends
    """
    total_steps = z.attrs.get('Steps')
    agent_steps = np.array(z.attrs.get('AgentSteps'))

    ids = np.empty(total_steps, dtype=np.int)
    current_step = 0

    for steps in agent_steps:
        end = min(current_step + steps, total_steps)
        # we always end at step 1 as the final state is not in the datasets
        ids[current_step:end] = np.arange(end - current_step, 0, -1)
        current_step += steps

    return ids


def get_agent_died_in_step(single_episode_actions, single_episode_dead):
    """
    For each agent, get the step in which it died. 0 if it is still alive.

    :param single_episode_actions: The actions of all agents in this episode.
    :param single_episode_dead: The final result of the episode.
    :return: array of steps in which the agents died
    """
    died_in_step = np.empty(4, dtype=int)
    for id, actions in enumerate(single_episode_actions):
        died_in_step[id] = len(actions) if single_episode_dead[id] else 0

    return died_in_step


def get_value_target(z, value_version: int, discount_factor: float, mcts_val_weight: Optional[float]) -> np.ndarray:
    """
    Creates the value target for a zarr dataset z by combining episode values (according to the given value_version)
    with value predictions from the dataset.

    :param z: The zarr dataset
    :param value_version: Specifies how the value is defined.
    <list>
    <li>1 = considers only win/loss</li>
    <li>2 = considers defeated agents</li>
    <li>3 = similar to 2 but with intermediate rewards</li>
    <li>4 = similar to 2 but with less punishment for dying. Focuses on number of dead opponents.</li>
    </list>
    :param discount_factor: The discount factor for the episode values (not mcts values)
    :param mcts_val_weight: Static weight of mcts values (completely ignored when None)
        val_target = mcts_val_weight * mcts values + (1 - mcts_val_weight) * episode values
        Note that episode values can contain discounting towards the mcts values (or 0 if mcts_val_weight is None)
    :return: The value target for z
    """
    assert 0 <= discount_factor <= 1, f"Invalid value for discount factor {discount_factor}"
    assert mcts_val_weight is None or 0 <= mcts_val_weight <= 1, f"Invalid value for mcts value weight {mcts_val_weight}"

    total_steps = z.attrs.get('Steps')
    agent_steps = np.array(z.attrs.get('AgentSteps'))
    agent_ids = np.array(z.attrs.get('AgentIds'))
    agent_episode = np.array(z.attrs.get('AgentEpisode'))
    episode_winner = np.array(z.attrs.get('EpisodeWinner'))
    episode_dead = np.array(z.attrs.get('EpisodeDead'))
    episode_actions = z.attrs.get('EpisodeActions')
    # episode_draw = np.array(z.attrs.get('EpisodeDraw'))
    # episode_done = np.array(z.attrs.get('EpisodeDone'))

    all_mcts_q = np.array(z["q"])
    # replace nans of invalid actions
    all_mcts_q[all_mcts_q != all_mcts_q] = float('-inf')
    # get mcts values (= max q)
    all_mcts_val = np.max(all_mcts_q, axis=-1)

    if mcts_val_weight == 1:
        return all_mcts_val

    def get_combined_target(mcts_val, target_val, discounting_factors):
        if mcts_val_weight is None:
            return discounting_factors * target_val

        return (
            # static weight for the values from mcts
            mcts_val_weight * mcts_val
            # combine with target values
            + (1 - mcts_val_weight) * (
                # target values are discounted
                discounting_factors * target_val
                # for discounting = 0, the target is mcts_val and not 0
                + (1 - discounting_factors) * mcts_val
            )
        )

    val_target = np.empty(total_steps)
    current_step = 0
    for agent_ep_idx in range(0, len(agent_steps)):
        agent_id = agent_ids[agent_ep_idx]
        ep = agent_episode[agent_ep_idx]
        steps = agent_steps[agent_ep_idx]
        winner = episode_winner[ep]
        dead = episode_dead[ep][agent_id]

        died_in_step = get_agent_died_in_step(episode_actions[ep], episode_dead[ep])

        # min to handle cut datasets
        next_step = min(current_step + steps, total_steps)
        num_steps = next_step - current_step

        episode_discounting = np.power(discount_factor, np.arange(steps - 1, steps - 1 - num_steps, -1))
        episode_mcts_val = all_mcts_val[current_step:next_step]

        # TODO: Adapt for team mode
        if value_version == 1:
            # only distribute rewards when the (agent) episode is done
            if winner == agent_id:
                episode_value = 1
            elif dead:
                episode_value = -1
            else:
                # episode not done
                episode_value = 0

            episode_target = get_combined_target(episode_mcts_val, episode_value, episode_discounting)
        elif value_version == 2:
            # get number of opponents that died before our agent
            if dead:
                num_dead_opponents = (died_in_step[died_in_step != 0] <= died_in_step[agent_id]).sum() - 1
            else:
                num_dead_opponents = (died_in_step != 0).sum()

            episode_value = num_dead_opponents * 1.0 / 3 - dead
            episode_target = get_combined_target(episode_mcts_val, episode_value, episode_discounting)
        elif value_version == 3:
            episode_target = np.zeros(num_steps)

            for id, step in enumerate(died_in_step):
                if step != 0:
                    mcts_val = episode_mcts_val[0:step +1]
                    discounting = episode_discounting[-(step + 1):]
                    if id == agent_id:
                        episode_target[0:step+1] = get_combined_target(mcts_val, -1, discounting)
                    else:
                        episode_target[0:step+1] = get_combined_target(mcts_val, 1.0 / 3, discounting)
        elif value_version == 4:
            # get number of opponents that died before our agent
            if dead:
                num_dead_opponents = (died_in_step[died_in_step != 0] <= died_in_step[agent_id]).sum() - 1
            else:
                num_dead_opponents = (died_in_step != 0).sum()

            episode_value = -1 + 4.0 / 7 * num_dead_opponents + (0 if dead else 2.0 / 7)
            episode_target = get_combined_target(episode_mcts_val, episode_value, episode_discounting)
        else:
            raise ValueError(f"Unknown value version {value_version}")

        # calculate discount factors backwards
        val_target[current_step:next_step] = episode_target
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


def create_data_loaders(path_infos: Union[str, List[Union[str, Tuple[str, float]]]], value_version: int,
                        discount_factor: float, mcts_val_weight: Optional[float], test_size: float, batch_size: int,
                        batch_size_test: int, train_transform = None,verbose: bool = True, sequence_length=None,
                        num_workers=2, only_test_last=False, train_sampling_mode: str = 'complete'
                        ) -> [DataLoader, DataLoader]:
    """
    Returns pytorch dataset loaders for a given path

    :param path_infos: The path information of the zarr datasets which should be used. Expects a single path or a list
                      containing strings (paths) or a tuple of the form (path, proportion) where 0 <= proportion <= 1
                      is the number of samples which will be selected randomly from this data set.
    :param value_version: The value version that should be used
    :param discount_factor: The discount factor that should be used
    :param mcts_val_weight: Weight for mcts values (None if only episode values should be used)
    :param test_size: Percentage of data to use for testing
    :param batch_size: Batch size to use for training
    :param batch_size_test: Batch size to use for testing
    :param train_transform: Data transformation for train data loading
    :param verbose: Log debug information
    :param sequence_length: Sequence length used in the train loader
    :param num_workers: The number of workers used for loading
    :param only_test_last: Whether only the last dataset is used for testing
    :param train_sampling_mode: Defines how the samples are chosen. Possible values: <br>
        <list>
        <li>'complete' to load all samples in random order.</li>
        <li>'weighted_steps_to_end' to assign exponentially decreasing weights to each sample based on the
        number of steps until the individual episode ends. Samples are chosen with replacement using the
        the normalized weights as probabilities.
        </list>
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

    def get_test_size(path_index):
        if only_test_last:
            if path_index == len(path_infos) - 1:
                return test_size
            else:
                return 0
        else:
            return test_size

    def get_total_sample_count():
        all_train_samples = 0
        all_test_samples = 0

        for i, info in enumerate(path_infos):
            path, proportion = get_elems(info)
            z = zarr.open(str(path), 'r')
            num_samples = int(z.attrs['Steps'] * proportion)

            test_samples = int(num_samples * get_test_size(i))
            train_samples = num_samples - test_samples

            all_train_samples += train_samples
            all_test_samples += test_samples

        return all_train_samples, all_test_samples

    total_train_samples, total_test_samples = get_total_sample_count()

    if verbose:
        print(f"Loading {total_train_samples + total_test_samples} samples from {len(path_infos)} dataset(s) with "
              f"test size {test_size}{' only last' if only_test_last else ''} ({total_test_samples} samples)")

    data_train = PommerDataset.create_empty(total_train_samples, transform=train_transform,
                                            sequence_length=sequence_length, return_ids=(sequence_length is not None))
    data_test = PommerDataset.create_empty(total_test_samples, return_ids=(sequence_length is not None))

    # create a container for all samples
    if verbose:
        print(f"Created containers with (train: {len(data_train)}, test: {len(data_test)}) samples "
              f"and train sequence length {sequence_length}")

    buffer_train_idx = 0
    buffer_test_idx = 0
    for i, info in enumerate(path_infos):
        path, proportion = get_elems(info)
        elem_samples = PommerDataset.from_zarr_path(path, value_version, discount_factor, mcts_val_weight,
                                                    verbose=verbose)

        if verbose:
            print(f"> Loading '{path}' with proportion {proportion}")

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

        test_nb = int(elem_samples_nb * get_test_size(i))
        train_nb = elem_samples_nb - test_nb

        data_train.set(elem_samples, buffer_train_idx, elem_samples_from, train_nb)
        data_test.set(elem_samples, buffer_test_idx, elem_samples_from + train_nb, test_nb)
        del elem_samples

        # copy first num_samples samples
        if verbose:
            print("Copied {} ({}, {}) samples ({:.2f}%) to buffers @ ({}, {})"
                  .format(elem_samples_nb, train_nb, test_nb, proportion * 100, buffer_train_idx, buffer_test_idx))

        buffer_train_idx += train_nb
        buffer_test_idx += test_nb

    assert buffer_train_idx == total_train_samples and buffer_test_idx == total_test_samples, \
        f"The number of copied samples is wrong.. " \
        f"{(buffer_train_idx, total_train_samples, buffer_test_idx, total_test_samples)}"

    if verbose:
        print(f"Creating DataLoaders with train sampling mode {train_sampling_mode}..")

    if train_sampling_mode == 'complete':
        train_loader = DataLoader(data_train, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    elif train_sampling_mode == 'weighted_steps_to_end':
        train_weights = np.clip(np.power(0.97, data_train.steps_to_end - 1), 0.05, 1)
        sampler = WeightedRandomSampler(train_weights, len(data_train), replacement=True)
        train_loader = DataLoader(data_train, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
    elif train_sampling_mode == 'weighted_value_class':
        unique_values = list(np.unique(data_train.val))
        value_class_counts = {}
        value_class_weights = {}
        for value in unique_values:
            num_samples = (data_train.val == value).sum()
            value_class_counts[value] = num_samples
            value_class_weights[value] = (1.0 / len(unique_values)) * (1.0 / num_samples)

        if verbose:
            print(f"Value weighting with {len(unique_values)} classes")
            print(value_class_counts)
            print(value_class_weights)

        if discount_factor != 1:
            print("Warning: Value class weighting was created for discount factor 1.")

        train_weights = np.empty(len(data_train))
        for a in range(0, len(data_train)):
            train_weights[a] = value_class_weights[data_train.val[a]]

        sampler = WeightedRandomSampler(train_weights, len(data_train), replacement=True)
        train_loader = DataLoader(data_train, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
    else:
        raise ValueError(f"Unknown train_sampling_mode {train_sampling_mode}")

    test_loader = DataLoader(data_test, batch_size=batch_size_test, num_workers=num_workers) if total_test_samples > 0 else None

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
