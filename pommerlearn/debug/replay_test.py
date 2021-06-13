import zarr

from env.replay_env import PommeReplay


def main():
    z = zarr.open('data_0.zr', 'r')

    # last episode can be cut off
    full_episodes = len(z.attrs['EpisodeSteps']) - 1
    correct_results = 0

    shortest_incorrect = None
    shortest_incorrect_steps = 0

    for ep in range(0, full_episodes):
        result_is_correct = PommeReplay.play(z, ep)
        print("Episode {}: Steps {}, correct result {}".format(ep, z.attrs['EpisodeSteps'][ep], result_is_correct))

        if result_is_correct:
            correct_results += 1
        else:
            if shortest_incorrect is None:
                shortest_incorrect = ep
                shortest_incorrect_steps = z.attrs['EpisodeSteps'][ep]
            else:
                steps = z.attrs['EpisodeSteps'][ep]
                if steps < shortest_incorrect_steps:
                    shortest_incorrect_steps = steps
                    shortest_incorrect = ep

    print("Total: {}/{} ({:.2f}%) were correct".format(correct_results, full_episodes, correct_results/full_episodes * 100))

    if shortest_incorrect is not None:
        print("Ep {}, steps {}".format(shortest_incorrect, shortest_incorrect_steps))
        PommeReplay.play(z, shortest_incorrect, render=True, render_pause=None)


if __name__ == '__main__':
    main()
