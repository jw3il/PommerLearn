import pommerman
import numpy as np
import time

import util


def ffa_eval_pooled(agent_classes, episodes, verbose):
    """
    Evaluates the given agents in the Free For All (FFA) pommerman environment using parallel workers.

    :param agent_classes: The classes of the individual agents (has to contain four elements)
    :param episodes: The number of episodes
    :param verbose: Whether to print verbose status information
    :return: The results of the evaluation of shape (episodes, 5) where the first column [:, 0] contains the result
             of the match (tie, win, incomplete) and the remaining columns contain the individual (final) rewards.
    """
    def status(tmp_res):
        if verbose:
            print("Completed {}/{} episodes.".format(len(tmp_res), episodes))

    # gather the results of ..
    results = np.concatenate(
        # .. "episode" jobs for the free for all evaluation
        util.pooled_multi_run(ffa_eval, [agent_classes, 1, False, False], episodes, status_fun=status)
    )

    if verbose:
        ffa_print_stats(results, episodes)

    return results


def ffa_eval(agent_classes, episodes, verbose, visualize):
    """
    Evaluates the given agents in the Free For All (FFA) pommerman environment.

    :param agent_classes: The classes of the individual agents (has to contain four elements)
    :param episodes: The number of episodes
    :param verbose: Whether to print verbose status information
    :param visualize: Whether to visualize the execution
    :return: The results of the evaluation of shape (episodes, 5) where the first column [:, 0] contains the result
             of the match (tie, win, incomplete) and the remaining columns contain the individual (final) rewards.
    """
    # Create a set of agents (exactly four)
    agent_list = [
        agent_classes[0](),
        agent_classes[1](),
        agent_classes[2](),
        agent_classes[3](),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    # first element: result, additional elements: rewards
    steps = np.empty(episodes)
    results = np.empty((episodes, 1 + len(agent_list)))

    start = time.time()

    # Run the episodes just like OpenAI Gym
    for i_episode in range(episodes):
        state = env.reset()
        done = False
        reward = []
        info = {}
        step = 0
        while not done:
            if visualize:
                env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            step += 1

        steps[i_episode] = step

        result = info['result']
        # save the result
        results[i_episode, 0] = result.value
        results[i_episode, 1:] = reward

        if verbose:
            delta = time.time() - start
            print('\r{:.2f} sec > Episode {} finished with {} ({}) | Stats: {}'.format(
                delta, i_episode, result, reward, ffa_get_stats_inline(results, i_episode + 1)
            ))
            if i_episode % 10 == 9:
                ffa_print_stats(results, steps, i_episode + 1)

    env.close()

    if verbose:
        delta = time.time() - start
        print("Total time: {:.2f} sec".format(delta))
        ffa_print_stats(results, steps, episodes)

    return results


def ffa_print_stats(results, steps, episodes):
    num_won, num_ties = ffa_get_stats(results, episodes)

    print("Evaluated {} episodes".format(episodes))
    print("Average steps: {}".format(steps[:episodes].mean()))

    total_won = np.sum(num_won)
    print("Wins: {} ({:.2f}%)".format(total_won, total_won / episodes * 100))
    for a in range(len(num_won)):
        print("> Agent {}: {} ({:.2f}%)".format(a, num_won[a], num_won[a] / total_won * 100))

    num_ties = np.sum(results[:, 0] == pommerman.constants.Result.Tie.value)
    print("Ties: {} ({:.2f}%)".format(num_ties, num_ties / episodes * 100))


def ffa_get_stats_inline(results, episodes):
    num_won, num_ties = ffa_get_stats(results, episodes)

    return "{} ({})".format(num_won, num_ties)


def ffa_get_stats(results, episodes):
    # Count how often each agent achieved a final reward of "1"
    num_won = np.sum(results[0:episodes, 1:] == 1, axis=0)
    # In a tie, every player receives -1 reward
    num_ties = np.sum(results[0:episodes, 0] == pommerman.constants.Result.Tie.value)

    assert np.sum(num_won) + num_ties == episodes

    return num_won, num_ties
