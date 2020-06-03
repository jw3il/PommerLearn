import z5py
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


def main():
    # Note: You have to create that dataset first
    f = z5py.File('data.zr', use_zarr_format=True)
    print("Container content: {} episodes".format(len(f) / 2))

    print("Printing first episode info")
    ep_0_obs = f['data_ep0_obs']
    ep_0_act = f['data_ep0_act']

    steps = len(ep_0_act)
    print("Steps: {} (consistent: {})".format(steps, len(ep_0_act) == len(ep_0_obs)))

    print("Actions")
    print(ep_0_act[:])

    print("Observations of first step:")
    print("Shape", ep_0_obs[0].shape)
    # plot_obs(ep_0_obs[0])

    print("Creating gif of episode 0")
    save_episode_obs_gif(ep_0_obs, "./episode_0.gif")


if __name__ == '__main__':
    main()
