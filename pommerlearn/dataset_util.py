import numpy as np
from pommerman import constants
import zarr

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

