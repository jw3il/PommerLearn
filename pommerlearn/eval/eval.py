import argparse
import pommerman
import pommerman.agents as agents
from pypomcpp.autocopy import AutoCopy
from pypomcpp.cppagent import CppAgent
from pypomcpp.util import evaluate
from env.env_rand_positions import PommeRandomPositon
from training.util_argparse import check_dir, check_file

def get_agent_list(autolib: AutoCopy, inputs):
    """Provides different exemplary agent configurations"""
    
    # docker images have to be pulled once
    # They can be found at https://hub.docker.com/u/multiagentlearning
    docker_img = {
        "dypm": "multiagentlearning/dypm.1",
        "hazoj": "multiagentlearning/hakozakijunctions", # Winner NeurIPD2018
        "gorog": "multiagentlearning/nips19-gorogm.gorogm", # Winner NeurIP2019
        "skynet": "multiagentlearning/skynet955",
        "navocado": "multiagentlearning/navocado"
    }

    def create_agent(name, port=15000):
        """Create an agent given its name and the cli inputs"""
        if name == 'crazyara':
            agent_name = f"CrazyAraAgent:{inputs.model_dir}:{inputs.state_size}:{inputs.simulations}:{inputs.movetime}"
            if inputs.virtual_step:
                agent_name += ':virtualStep'
            if inputs.terminal:
                agent_name += ':mctsSolver'
            agent = CppAgent(autolib(), agent_name, 42, False)

        elif name == 'rawnet':
            agent_name = f"RawNetAgent:{inputs.model_dir}:{inputs.state_size}"
            if inputs.virtual_step:
                agent_name += ':virtualStep'
            agent = CppAgent(autolib(), agent_name)   

        elif name == 'simple':
            agent = CppAgent(autolib(), "SimpleAgent", seed=17)

        elif name in docker_img.keys():
            agent = agents.DockerAgent(docker_image=docker_img[name], port=port)

        else:
            print("Unkown opponent agent type")
            raise NotImplementedError

        return agent

    port = 13_000
    # create a team env with 2 Crazy Ara agents and 2 opponents
    if inputs.env == 'team':
        agent_list = [
            create_agent('crazyara'),
            create_agent(inputs.opponent, port),
            create_agent('crazyara'),
            create_agent(inputs.opponent, port+1)
        ]
    # create a team env with a Crazy Ara agents and 3 opponents
    else:
        agent_list = [
            create_agent('crazyara', inputs),
            create_agent(inputs.opponent, port),
            create_agent(inputs.opponent, port+1),
            create_agent(inputs.opponent, port+2)
        ]

    return agent_list


def parse_args():
    """CLI parser

    :return: parsed arguments
    """

    parser = argparse.ArgumentParser(description='PommerLearn RL Loop')

    # RawNet/Crazy Ara Parameters
    parser.add_argument('--model_dir', type=check_dir, help="Directory of the agent.")
    parser.add_argument('--state_size', type=int, default=0, help='state size')
    # Crazy Ara Parameters
    parser.add_argument('--simulations', type=int, default=100, help='number of simulations')
    parser.add_argument('--movetime', type=int, default=100, help='move time')
    parser.add_argument('--virtual_step', type=int, default=100, help='use virtual step')
    parser.add_argument('--terminal', type=int, default=100, help='use mctsSolver')
    # Opponents
    parser.add_argument('-o', '--opponent', type=str, default="simple", help='name of the opponent agent possible [crazyara, rawnet,  simple, dypm, hazoj, gorog, skynet, navocado]')

    # Evaluation Parameters
    parser.add_argument('--eval_path', type=str, help='path to folder that will be created for storing the evaluation results. If not specified a new folder will be created in ./eval_plotting/')
    parser.add_argument('--lib', type=str, default="./build/libPommerLearnPy.so", help='libpath')
    parser.add_argument('-g', '--games', type=int, default=10, help='number of games')
    parser.add_argument('-e', '--env', type=str, default="team", help='game mode: [ffa, ffa_random, team]')

    parser.add_argument('--use_true_state', type=bool, default=False, help='Whether to use the true state instead of (partial) observations')

    parsed_args = parser.parse_args()
    return parsed_args


def main():
    """ run evaluation and save results """
    inputs = parse_args()

    if inputs.model_dir is None: 
        raise ValueError('inputs.model_dir must be set to the agents model directory')

    # path to save results (if none -> creates folder and saves under PommerLearn/eval_plotting/"Pomme_e_{games}")
    eval_path = inputs.eval_path

    # Automatically copy the library to allow instantiating multiple cpp agents
    autolib = AutoCopy(inputs.lib, "./libpomcpp_tmp")

    # create Agents

    agent_list = get_agent_list(autolib, inputs)

    # Make the environment using the agent list
    if inputs.env == 'team':
        env_type ='PommeTeamCompetition-v0'
        env = pommerman.make(env_type, agent_list)
    elif inputs.env == 'ffa':
        env_type ='PommeFFACompetition-v0'
        env = pommerman.make(env_type, agent_list)  #PommeRandomPositon(agent_list) - for ffa version with random starting positions
    elif inputs.env == 'ffa_random':
        env_type ='PommeFFACompetition-v0'
        env = PommeRandomPositon(agent_list) # for ffa version with random starting positions

    if inputs.use_true_state:
        for a in agent_list:
            if isinstance(a, CppAgent):
                a.use_env_state(env)

    try:
        evaluate(env, inputs.games, verbose=True, visualize=False, stop=False, individual_plots=True, plot_agents_alive=False, eval_save_path=eval_path, log_json=True, env_type=env_type)
    finally:
        autolib.delete_copies()


if __name__ == "__main__":
    main()
