import time

import numpy as np
import pommerman.envs.v0 as v0
from pommerman import utility, constants, agents
from pommerman.constants import *

from dataset_util import get_agent_actions


class PommeReplay(v0.Pomme):
    def __init__(self, board_size=constants.BOARD_SIZE):
        env_kwargs = {
            'game_type': constants.GameType.FFA,
            'board_size': board_size,
            'max_steps': constants.MAX_STEPS,
            'render_fps': 24,
            'env': 'ReplayEnv',
        }
        super().__init__(**env_kwargs)

        self._step_count = None
        self._board = None
        self._items = None
        self._bombs = None
        self._flames = None
        self._powerups = None

        # initialize agents which do nothing, they are just containers for a part of the game state
        # and will throw errors when you try to actually retrieve actions from them
        agent_list = [
            agents.BaseAgent(),
            agents.BaseAgent(),
            agents.BaseAgent(),
            agents.BaseAgent()
        ]
        for id_, agent in enumerate(agent_list):
            agent.init_agent(id_, env_kwargs['game_type'])

        self.set_agents(agent_list)

    def reset_to_initial_state(self, str_init_state, default_ammo=1, default_blast_strength=1, default_can_kick=False):
        """
        Initializes (resets) the game state with the given string representation of the initial state.

        :param str_init_state: The string representation of the initial state
        :param default_ammo: The default ammo of the agents
        :param default_blast_strength: The default blast strength of the agents
        :param default_can_kick: Whether agents can already kick bombs when the episode starts
        """

        def agent_dict_entry(agent_id, agent_row, agent_col):
            return {
                'agent_id': agent_id,
                'is_alive': True,
                'position': (agent_row, agent_col),
                'ammo': default_ammo,
                'blast_strength': default_blast_strength,
                'can_kick': default_can_kick
            }

        board = np.empty((self._board_size, self._board_size)).astype(np.uint8) * Item.Passage.value
        items = {}

        col = 0
        row = 0
        for c in str_init_state:
            if c == '0':
                board[row, col] = Item.Passage.value
            elif c == '1':
                board[row, col] = Item.Rigid.value
            elif c == '2':
                board[row, col] = Item.Wood.value
            elif c == '3':
                board[row, col] = Item.Wood.value
                items[(row, col)] = Item.ExtraBomb.value
            elif c == '4':
                board[row, col] = Item.Wood.value
                items[(row, col)] = Item.IncrRange.value
            elif c == '5':
                board[row, col] = Item.Wood.value
                items[(row, col)] = Item.Kick.value
            elif c == 'A':
                board[row, col] = Item.Agent0.value
            elif c == 'B':
                board[row, col] = Item.Agent1.value
            elif c == 'C':
                board[row, col] = Item.Agent2.value
            elif c == 'D':
                board[row, col] = Item.Agent3.value
            else:
                print("Unknown board item {}".format(c))

            col = col + 1
            if col >= self._board_size:
                col = 0
                row = row + 1

        self._step_count = 0
        self._board = board
        self._items = items
        self._bombs = []
        self._flames = []
        self._powerups = []

        for agent_id, agent in enumerate(self._agents):
            pos = np.where(self._board == utility.agent_value(agent_id))
            row = pos[0][0]
            col = pos[1][0]
            agent.set_start_position((row, col))
            agent.reset()

    @staticmethod
    def play(z, episode, render=False, render_pause=0.1, verbose_result_check=True):
        """
        Plays an episode from the given zarr dataset. Also checks whether the results are consistent.

        :param z: The zarr dataset
        :param episode: The episode index
        :param render: Whether to render the playback. Waits for keyboard input when None.
        :param verbose_result_check: Whether to display verbose information about the result inconsistencies
        :return Whether the expected result is equal to the result of the replay.
        """

        replay = PommeReplay()
        # first load the initial state from the metadata
        replay.reset_to_initial_state(z.attrs['EpisodeInitialState'][episode])
        # then get all actions from the zarr dataset
        actions = get_agent_actions(z, episode)

        if render:
            replay.render(do_sleep=False)
            if render_pause is None:
                input()
            else:
                time.sleep(render_pause)

        done, info = None, None
        for step in range(0, len(actions)):
            # execute the actions step by step
            state, reward, done, info = replay.step(actions[step, :])
            if render:
                replay.render(do_sleep=False)
                if render_pause is None:
                    input()
                else:
                    time.sleep(render_pause)

        expected_winner = z.attrs['EpisodeWinner'][episode]
        py_result = info['result']

        if expected_winner == -1:
            if not (py_result == constants.Result.Tie or py_result == constants.Result.Incomplete):
                if verbose_result_check:
                    print("Episode {}: Results do not match! Result of python env was {} but expected winner was {}"
                          .format(episode, py_result, expected_winner))
                return False
        else:
            if 'winners' in info:
                if not (expected_winner in info['winners'] and len(info['winners']) == 1):
                    if verbose_result_check:
                        print("Episode {}: Results do not match! Winner of python env was {} but expected winner was {}"
                              .format(episode, info['winners'], expected_winner))
                    return False
            else:
                if verbose_result_check:
                    print("Episode {}: Results do not match! Result of python env was {} but expected winner was {}"
                          .format(episode, py_result, expected_winner))
                return False

        return True
