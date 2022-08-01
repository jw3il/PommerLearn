import pommerman
import pommerman.agents as agents
from pommerman.agents.simple_agent import SimpleAgent
from pypomcpp.cppagent import CppAgent
from pypomcpp.util import evaluate
from shutil import copyfile

lib_path = "./libPommerLearnPy.so"


def create_lib_copy():
    if hasattr(create_lib_copy, "calls"):
        create_lib_copy.calls += 1
    else:
        create_lib_copy.calls = 0

    local_lib_path = f"./local_lib_copy_{create_lib_copy.calls}.so"
    copyfile(lib_path, local_lib_path)
    return local_lib_path


model_path="./stateless_model/onnx"
state_size=0
simulations=100
movetime=100

agent_list = [
    CppAgent(create_lib_copy(), f"CrazyAraAgent:{model_path}:{state_size}:{simulations}:{movetime}"),
    CppAgent(create_lib_copy(), f"RawNetAgent:{model_path}:{state_size}"),
    CppAgent(create_lib_copy(), "SimpleUnbiasedAgent", seed=16),
    # CppAgent(create_lib_copy(), "SimpleAgent", seed=17),
    agents.SimpleAgent()
]

# Make the "Free-For-All" environment using the agent list
env = pommerman.make('PommeFFACompetition-v0', agent_list)

USE_ENV_STATE = False

if USE_ENV_STATE:
    for a in agent_list:
        if isinstance(a, CppAgent):
            a.use_env_state(env)

evaluate(env, 10, True, False)
