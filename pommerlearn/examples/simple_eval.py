import pommerman.agents as agents
import util_eval


def main():
    agent_classes = [
        agents.SimpleAgent,
        agents.SimpleAgent,
        agents.SimpleAgent,
        agents.SimpleAgent
    ]
    util_eval.ffa_eval_pooled(agent_classes, episodes=100, verbose=True)

    agent_classes[0] = agents.RandomAgent
    util_eval.ffa_eval_pooled(agent_classes, episodes=100, verbose=True)


if __name__ == '__main__':
    main()
