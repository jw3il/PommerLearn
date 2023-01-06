import pommerman
import pommerman.agents as agents
from pypomcpp.autocopy import AutoCopy
from pypomcpp.cppagent import CppAgent
from pypomcpp.util import evaluate, print_stats
from shutil import copyfile
from env.env_rand_positions import PommeRandomPositon

lib_path = "./build/libPommerLearnPy.so"
model_path="build/model/onxx"

def get_agent_list(index, autolib: AutoCopy, state_size, simulations, movetime):
    # Provides different exemplary agent configurations
    
    # docker images have to be pulled once
    # They can be found at https://hub.docker.com/u/multiagentlearning
    docker_img = {
        "dypm": "multiagentlearning/dypm.1",
        "hazoj": "multiagentlearning/hakozakijunctions", # Winner NeurIPD2018
        "gorog": "multiagentlearning/nips19-gorogm.gorogm", # Winner NeurIP2019
        "skynet": "multiagentlearning/skynet955",
        "navocado": "multiagentlearning/navocado"
    }

    # vs Docker Agents
    if index == 0:
        port = 15000
        img = docker_img['gorog']
        agent_list = [
            CppAgent(autolib(), f"CrazyAraAgent:{model_path}:{state_size}:{simulations}:{movetime}:virtualStep",42,False),
            agents.DockerAgent(docker_image=img, port=port+1),
            CppAgent(autolib(), f"CrazyAraAgent:{model_path}:{state_size}:{simulations}:{movetime}:virtualStep",42,False),
            agents.DockerAgent(docker_image=img, port=port+3),
        ]
    # CrazyAraAgent vs Simple
    elif index == 1:
        agent_list = [
            CppAgent(autolib(), f"CrazyAraAgent:{model_path}:{state_size}:{simulations}:{movetime}",42,False),
            CppAgent(autolib(), "SimpleAgent", seed=17),
            CppAgent(autolib(), "SimpleAgent", seed=18),
            CppAgent(autolib(), "SimpleAgent", seed=19),
            
        ]
    # self-play & different #simulations
    elif index == 2:
        agent_list = [
            CppAgent(autolib(), f"CrazyAraAgent:{model_path}:{state_size}:{simulations}:{movetime}"),
            CppAgent(autolib(), f"CrazyAraAgent:{model_path}:{state_size}:{simulations+50}:{movetime}"),
            CppAgent(autolib(), f"CrazyAraAgent:{model_path}:{state_size}:{simulations+250}:{movetime}"),
            CppAgent(autolib(), f"CrazyAraAgent:{model_path}:{state_size}:{simulations+500}:{movetime}"),
            
        ]
    # Simple Agents only
    else:
        agent_list = [
            CppAgent(autolib(), "SimpleAgent", seed=16),
            CppAgent(autolib(), "SimpleAgent", seed=17),
            CppAgent(autolib(), "SimpleAgent", seed=18),
            CppAgent(autolib(), "SimpleAgent", seed=19),
        ]
    return agent_list

def main():
    state_size=0
    simulations=100
    movetime=100
    games = 10

    # path to save results (if none -> creates folder and saves under PommerLearn/eval_plotting/"Pomme_e_{games}")
    eval_path = None

    # Automatically copy the library to allow instantiating multiple cpp agents
    autolib = AutoCopy(lib_path, "./libpomcpp_tmp")

    agent_list = get_agent_list(1, autolib, state_size, simulations, movetime)

    # Make the "Free-For-All" environment using the agent list
    env_type='PommeTeamCompetition-v0'
    env = pommerman.make(env_type, agent_list)  #PommeRandomPositon(agent_list) - for ffa version with random starting positions

    use_env_state = False

    if use_env_state:
        for a in agent_list:
            if isinstance(a, CppAgent):
                a.use_env_state(env)

    try:
        eval_path = None#"./eval_plotting/gorog0"
        evaluate(env, games, verbose=True, visualize=False, stop=False, individual_plots=True, plot_agents_alive=False, eval_save_path=eval_path, log_json=True, env_type=env_type)
    finally:
        autolib.delete_copies()


if __name__ == "__main__":
    main()
