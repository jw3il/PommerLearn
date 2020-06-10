import zarr
import numpy as np
import matplotlib.pyplot as plt
import math
import imageio


def create_obs_fig(obs, show_plane_title=True):
    num_planes = obs.shape[0]
    figsize_side = math.ceil(math.sqrt(num_planes))

    fig = plt.figure(figsize=(figsize_side, figsize_side))

    # show every plane
    for i in range(0, num_planes):
        fig.add_subplot(figsize_side, figsize_side, i+1)

        if show_plane_title:
            plt.title("Plane {}".format(i + 1))

        plt.imshow(obs[i, :, :], origin='lower', vmin=0, vmax=1)

    return fig


def plot_obs(obs):
    create_obs_fig(obs)
    plt.show()
    plt.clf()
    plt.cla()


def render_obs_step_image(obs, step):
    fig = create_obs_fig(obs, False)
    fig.suptitle("Step: {}".format(step))

    # clear axes for cleaner image
    for ax in fig.axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig.canvas.draw()

    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image


# see https://ndres.me/post/matplotlib-animated-gifs-easily/
def save_episode_obs_gif(obs_array, filename, verbose=True, fps=10):
    steps = obs_array.shape[0]

    if verbose:
        print("Step..", end='')

    def gif_creation_step(step):
        if verbose:
            print("\rStep {}/{}".format(step, steps), end='')

        return render_obs_step_image(obs_array[step], step)

    imageio.mimsave(filename, [gif_creation_step(step) for step in range(steps)], fps=fps)

    if verbose:
        print("\rDone")


def get_agent_episode_slice(attrs, agent_episode):
    # sum up all steps up to our episode
    start_index = int(np.sum(attrs['AgentSteps'][0:agent_episode]))
    # add the amount of steps of the episode
    end_index = start_index + attrs['AgentSteps'][agent_episode]
    return slice(start_index, end_index)


def last_episode_is_cut(f):
    return np.sum(f.attrs.get('AgentSteps')) != f.attrs.get('Steps')


def main():
    # Note: You have to create that dataset first
    f = zarr.open('data_0.zr', 'r')

    print("Info:")
    print(f.info)

    attrs = f.attrs.asdict()
    print("Attributes: {}".format(attrs.keys()))

    dataset_size = len(f['act'])
    actual_steps = attrs['Steps']
    print("Dataset size: {}, actual steps: {}".format(dataset_size, actual_steps))

    print("Environment runs: {}, total agent episodes: {}, last episode is cut {}"
          .format(len(attrs['EpisodeSteps']), len(attrs['AgentSteps']), last_episode_is_cut(f)))

    agent_episode = 0
    episode_slice = get_agent_episode_slice(attrs, agent_episode)

    obs = f['obs'][episode_slice]
    act = f['act'][episode_slice]
    val = f['val'][episode_slice]

    # Warning: The last episode may have been cut
    obs_len = len(obs)
    agent_len = attrs['AgentSteps'][agent_episode]
    if obs_len == agent_len:
        print("Steps: {} = {}".format(obs_len, agent_len))
    else:
        print("Steps: {} != {} -- is last episode {}"
              .format(obs_len, agent_len, agent_episode == -1 or agent_episode == len(attrs['AgentSteps']) - 1))

    print("Actions")
    print(act)
    print("Values:")
    print(val)
    print("Observations of first step:")
    print("Shape", obs[0].shape)
    # plot_obs(ep_0_obs[0])

    print("Creating gif of observations")
    save_episode_obs_gif(obs, "./episode_obs.gif")


if __name__ == '__main__':
    main()
