"""
Example: Replay a single episode.
"""

import zarr
import numpy as np
from pommerlearn.env.replay_env import PommeReplay

# load dataset
z = zarr.open('data_0.zr')

# select an episode, e.g. one where agent 0 has won
winners = np.array(z.attrs.get('EpisodeWinner'))
agent_episode = np.argwhere(winners == 0)[0][0]

# print episode information
print(
    f"Agent episode {agent_episode}, "
    f"agent id {z.attrs.get('AgentIds')[agent_episode]}, "
    f"winner: {z.attrs.get('EpisodeWinner')[agent_episode]}, "
    f"steps: {z.attrs.get('AgentSteps')[agent_episode]}",
    f"dead: {np.array(z.attrs.get('EpisodeDead'))[agent_episode]}"
)

# start replay
PommeReplay.play(z, agent_episode, render=True, render_pause=0.2, verbose_policy=True)
