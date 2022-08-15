from tabnanny import verbose
import pommerman
import pommerman.agents as agents
from pommerman.agents.simple_agent import SimpleAgent
from pypomcpp.cppagent import CppAgent
from pypomcpp.util import evaluate, print_stats
from shutil import copyfile
from env.env_rand_positions import PommeRandomPositon

lib_path = "./build/libPommerLearnPy.so"


def create_lib_copy():
    if hasattr(create_lib_copy, "calls"):
        create_lib_copy.calls += 1
    else:
        create_lib_copy.calls = 0

    local_lib_path = f"./local_lib_copy_{create_lib_copy.calls}.so"
    copyfile(lib_path, local_lib_path)
    return local_lib_path

model_path="./build/models/mcts_data_500_60e_y0.97_w0.3_complete_2854c7e22a4b75539a74efa97a8166735f045356-20220719T102802Z-001/mcts_data_500_60e_y0.97_w0.3_complete_2854c7e22a4b75539a74efa97a8166735f045356/onnx"
#model_path="./build/models/mcts_data_500_60e_y0.97_w0.3_weighted_actions_2854c7e22a4b75539a74efa97a8166735f045356-20220719T102804Z-001/mcts_data_500_60e_y0.97_w0.3_weighted_actions_2854c7e22a4b75539a74efa97a8166735f045356/onnx"
def get_agent_list(index, state_size=0, simulations=0, movetime=100):
    # vs Simple Agents
    if index == 0:
        agent_list = [
            CppAgent(create_lib_copy(), f"CrazyAraAgent:{model_path}:{state_size}:{simulations}:{movetime}"),
            CppAgent(create_lib_copy(), "SimpleAgent", seed=17),
            CppAgent(create_lib_copy(), "SimpleAgent", seed=17),
            CppAgent(create_lib_copy(), "SimpleAgent", seed=17),
        ]
    # different #simulations
    elif index == 1:
        agent_list = [
            CppAgent(create_lib_copy(), f"CrazyAraAgent:{model_path}:{state_size}:{simulations}:{movetime}"),
            CppAgent(create_lib_copy(), f"CrazyAraAgent:{model_path}:{state_size}:{simulations+50}:{movetime}"),
            CppAgent(create_lib_copy(), f"CrazyAraAgent:{model_path}:{state_size}:{simulations+250}:{movetime}"),
            CppAgent(create_lib_copy(), f"CrazyAraAgent:{model_path}:{state_size}:{simulations+500}:{movetime}"),
            
        ]
    # self-play
    elif index == 1:
        agent_list = [
            CppAgent(create_lib_copy(), f"CrazyAraAgent:{model_path}:{state_size}:{simulations}:{movetime}"),
            CppAgent(create_lib_copy(), f"CrazyAraAgent:{model_path}:{state_size}:{simulations+50}:{movetime}"),
            CppAgent(create_lib_copy(), f"CrazyAraAgent:{model_path}:{state_size}:{simulations+250}:{movetime}"),
            CppAgent(create_lib_copy(), f"CrazyAraAgent:{model_path}:{state_size}:{simulations+500}:{movetime}"),
            
        ]
    # Simple Agents only
    else:
        agent_list = [
            CppAgent(create_lib_copy(), "SimpleAgent", seed=16),
            CppAgent(create_lib_copy(), "SimpleAgent", seed=17),
            CppAgent(create_lib_copy(), "SimpleAgent", seed=18),
            CppAgent(create_lib_copy(), "SimpleAgent", seed=19),
        ]
    return agent_list

def main():
    state_size=0
    simulations=100
    movetime=100
    games = 1000

    agent_list = get_agent_list(0, state_size, simulations, movetime)

    # Make the "Free-For-All" environment using the agent list
    #env = pommerman.make('PommeFFACompetition-v0', agent_list)
    env = PommeRandomPositon(agent_list)

    use_env_state = False

    if use_env_state:
        for a in agent_list:
            if isinstance(a, CppAgent):
                a.use_env_state(env)

    evaluate(env, games, True, False)


if __name__ == "__main__":
    main()
