import itertools
import random
import numpy as np
from pommerman.envs import v0
from pommerman import configs, constants
from pommerman.utility import inaccessible_passages


def make_board(size, num_rigid=0, num_wood=0, num_agents=4):
    """Make the random but symmetric board. With random starting positions for the agents

    The numbers refer to the Item enum in constants. This is:
     0 - passage
     1 - rigid wall
     2 - wood wall
     3 - bomb
     4 - flames
     5 - fog
     6 - extra bomb item
     7 - extra firepower item
     8 - kick
     9 - skull
     10 - 13: agents

    Args:
      size: The dimension of the board, i.e. it's sizeXsize.
      num_rigid: The number of rigid walls on the board. This should be even.
      num_wood: Similar to above but for wood walls.

    Returns:
      board: The resulting random board.
    """

    def lay_wall(value, num_left, coordinates, board):
        '''Lays all of the walls on a board'''
        x, y = random.sample(coordinates, 1)[0]
        coordinates.remove((x, y))
        coordinates.remove((y, x))
        board[x, y] = value
        board[y, x] = value
        num_left -= 2
        return num_left

    def make(size, num_rigid, num_wood, num_agents):
        '''Constructs a game/board'''
        # Initialize everything as a passage.
        board = np.ones((size,
                         size)).astype(np.uint8) * constants.Item.Passage.value

        # Gather all the possible coordinates to use for walls.
        coordinates = set([
            (x, y) for x, y in \
            itertools.product(range(size), range(size)) \
            if x != y])

        # Set the players down. Exclude them from coordinates.
        # Agent0 is in top left. Agent1 is in bottom left.
        # Agent2 is in bottom right. Agent 3 is in top right.
        assert (num_agents % 2 == 0)

        if num_agents == 2:
            agents = [(1, 1), (size - 2, size - 2)]
            random.shuffle(agents)
            board[agents[0][0], agents[0][1]] = constants.Item.Agent0.value
            board[agents[1][0], agents[1][1]] = constants.Item.Agent1.value
            opos_ix = 0

        else:
            agents = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]
            random.shuffle(agents)
            opos_x = size-2 if agents[0][0] == 1 else 1
            opos_y = size-2 if agents[0][1] == 1 else 1
            opos_ix = agents.index((opos_x, opos_y))
            board[agents[0][0], agents[0][1]] = constants.Item.Agent0.value
            board[agents[1][0], agents[1][1]] = constants.Item.Agent1.value
            board[agents[2][0], agents[2][1]] = constants.Item.Agent2.value
            board[agents[3][0], agents[3][1]] = constants.Item.Agent3.value

        for position in agents:
            if position in coordinates:
                coordinates.remove(position)

        # Exclude breathing room on either side of the agents.
        for i in range(2, 4):
            coordinates.remove((1, i))
            coordinates.remove((i, 1))
            coordinates.remove((size - 2, size - i - 1))
            coordinates.remove((size - i - 1, size - 2))

            if num_agents == 4:
                coordinates.remove((1, size - i - 1))
                coordinates.remove((size - i - 1, 1))
                coordinates.remove((i, size - 2))
                coordinates.remove((size - 2, i))

        # Lay down wooden walls providing guaranteed passage to other agents.
        wood = constants.Item.Wood.value
        if num_agents == 4:
            for i in range(4, size - 4):
                board[1, i] = wood
                board[size - i - 1, 1] = wood
                board[size - 2, size - i - 1] = wood
                board[size - i - 1, size - 2] = wood
                coordinates.remove((1, i))
                coordinates.remove((size - i - 1, 1))
                coordinates.remove((size - 2, size - i - 1))
                coordinates.remove((size - i - 1, size - 2))
                num_wood -= 4

        # Lay down the rigid walls.
        while num_rigid > 0:
            num_rigid = lay_wall(constants.Item.Rigid.value, num_rigid,
                                 coordinates, board)

        # Lay down the wooden walls.
        while num_wood > 0:
            num_wood = lay_wall(constants.Item.Wood.value, num_wood,
                                coordinates, board)

        return board, agents, opos_ix

    assert (num_rigid % 2 == 0)
    assert (num_wood % 2 == 0)
    board, agents, opos_ix = make(size, num_rigid, num_wood, num_agents)

    # Make sure it's possible to reach most of the passages.
    while len(inaccessible_passages(board, agents)) > 4:
        board, agents, opos_ix = make(size, num_rigid, num_wood, num_agents)

    return board, opos_ix

class PommeRandomPositon(v0.Pomme):
    def __init__(self, agent_list):
        env_kwargs = configs.ffa_competition_env()['env_kwargs']
        super().__init__(**env_kwargs)

        for id_, agent in enumerate(agent_list):
            agent.init_agent(id_, env_kwargs['game_type'])
        self.set_agents(agent_list)
        self.set_init_game_state(None)
        self.opposite_position_counter = [0 for _ in range(3)]
    
    def make_board(self):
        self._board, opos_ix = make_board(self._board_size, self._num_rigid,
                                         self._num_wood, len(self._agents))
        self.opposite_position_counter[opos_ix-1] += 1