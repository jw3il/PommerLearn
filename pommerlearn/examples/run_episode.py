import pommerman.agents as agents
import util_eval


def main():
    agent_classes = [
        agents.SimpleAgent,
        agents.SimpleAgent,
        agents.SimpleAgent,
        agents.SimpleAgent
    ]
    util_eval.ffa_eval(agent_classes, episodes=1, verbose=True, visualize=True)


if __name__ == '__main__':
    main()
