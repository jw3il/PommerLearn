import pommerman.agents as agents
import util_eval


def main():
    # util_eval.ffa_eval(agents.SimpleAgent(), episodes=1, visualize=True)
    util_eval.ffa_eval(agents.SimpleAgent(), episodes=100)
    util_eval.ffa_eval(agents.RandomAgent(), episodes=100)


if __name__ == '__main__':
    main()
