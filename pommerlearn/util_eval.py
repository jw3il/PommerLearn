import pommerman
from pommerman import agents
import numpy as np


def ffa_eval(agent: agents.base_agent, enemy_class=agents.SimpleAgent, episodes=100, visualize=False):
    # Create a set of agents (exactly four)
    agent_list = [
        agent,
        enemy_class(),
        enemy_class(),
        enemy_class(),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    results = np.empty(episodes)
    rewards = np.empty((episodes, len(agent_list)))

    # Run the episodes just like OpenAI Gym
    for i_episode in range(episodes):
        state = env.reset()
        done = False
        reward = []
        info = {}
        while not done:
            if visualize:
                env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)

        result = info['result']
        # save the result
        results[i_episode] = result.value
        rewards[i_episode, :] = reward

        print('\r> Episode {} finished with {} ({}) | Stats: {}'.format(
            i_episode, result, reward, ffa_get_stats_inline(results, rewards, i_episode + 1)
        ))

    env.close()

    ffa_print_stats(results, rewards, episodes)


def ffa_print_stats(results, final_rewards, episodes):
    num_won, num_ties = ffa_get_stats(results, final_rewards, episodes)

    print("Evaluated {} episodes".format(episodes))

    total_won = np.sum(num_won)
    print("Wins: {} ({:.2f}%)".format(total_won, total_won / episodes * 100))
    for a in range(len(num_won)):
        print("> Agent {}: {} ({:.2f}%)".format(a, num_won[a], num_won[a] / total_won * 100))

    num_ties = np.sum(results == pommerman.constants.Result.Tie.value)
    print("Ties: {} ({:.2f}%)".format(num_ties, num_ties / episodes * 100))


def ffa_get_stats_inline(results, final_rewards, episodes):
    num_won, num_ties = ffa_get_stats(results, final_rewards, episodes)

    return "{} ({})".format(num_won, num_ties)


def ffa_get_stats(results, final_rewards, episodes):
    # Count how often each agent achieved a final reward of "1"
    num_won = np.sum(final_rewards[0:episodes, :] == 1, axis=0)
    # In a tie, every player receives -1 reward
    num_ties = np.sum(results[0:episodes] == pommerman.constants.Result.Tie.value)

    assert np.sum(num_won) + num_ties == episodes

    return num_won, num_ties
