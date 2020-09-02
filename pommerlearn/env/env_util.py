import json
import copy
from pommerman import utility, constants


def state_to_json(env):
    """
    Get a json representation of the current state. NOT the same as env.get_json_info because of
    simplicity reasons (instead of multiple dumps and json substrings, we only generate one json object).
    :param env The environment which contains the current state.
    """

    ret = {
        'board_size': env._board_size,
        'step_count': env._step_count,
        'board': env._board,
        'agents': env._agents,
        'bombs': env._bombs,
        'flames': env._flames,
        'items': [[k, i] for k, i in env._items.items()],
        'intended_actions': env._intended_actions
    }

    return json.dumps(ret, cls=utility.PommermanJSONEncoder)


def agent_vals_to_ids(agents):
    # Agent items have values 10, 11, 12, 13
    for i in reversed(range(0, len(agents))):
        if agents[i] == constants.Item.AgentDummy:
            del agents[i]
            continue

        agents[i] = agents[i].value - 10


def agent_ints_to_ids(agents):
    for i in reversed(range(0, len(agents))):
        if agents[i] == constants.Item.AgentDummy.value:
            del agents[i]
            continue

        agents[i] = agents[i] - 10


def observation_to_json(obs):
    obs = copy.deepcopy(obs)
    # fix items -> ids
    agent_ints_to_ids(obs["alive"])
    agent_vals_to_ids(obs["enemies"])

    # change teammate observation to array so that it is consistent with enemies
    teammate = obs["teammate"]
    obs["teammate"] = [] if teammate == constants.Item.AgentDummy else [teammate.value - 10]

    return json.dumps(obs, cls=utility.PommermanJSONEncoder)


def escape_parenthesis(s):
    return s.replace("\"", "\\\"")


def print_json(json_string):
    print(f"\"{escape_parenthesis(json_string)}\"")
