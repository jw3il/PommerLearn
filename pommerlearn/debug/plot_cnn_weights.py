import itertools
from pathlib import Path
from typing import Union

import matplotlib.axes
import numpy as np
import onnx, onnx.numpy_helper
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def plot_image_grid(data: np.ndarray, label: str, symmetric_cmap: bool, cmap):
    """
    Plots the given 3d data as an image grid.

    :param data: nd array with three dimensions (img, y, x)
    :param symmetric_cmap: whether the cmap should be symmetric around 0
    :returns: the figure
    """
    if symmetric_cmap:
        max_val = np.max(np.abs(data))
        min_val = -max_val
    else:
        max_val = np.max(data)
        min_val = np.min(data)

    num_img = data.shape[0]
    fig_dim = int(np.ceil(np.sqrt(num_img)))
    fig, axs = plt.subplots(fig_dim, fig_dim)
    fig.suptitle(f"{label}")

    # plot images
    for idx in range(fig_dim * fig_dim):
        x = idx % fig_dim
        y = idx // fig_dim

        axs[y, x].axis('off')
        if idx < num_img:
            im = axs[y, x].imshow(data[idx], vmin=min_val, vmax=max_val, cmap=cmap)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    return fig


def plot_first_layer(model_path: str, model_label: str, plot_dir: Union[Path, str]):
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    # print(onnx.helper.printable_graph(model.graph))

    weights = next(filter(lambda x: x.name == "163", model.graph.initializer))
    weights = onnx.numpy_helper.to_array(weights)
    # bias = next(filter(lambda x: x.name == "164", model.graph.initializer))
    # bias = onnx.numpy_helper.to_array(bias)

    if isinstance(plot_dir, str):
        plot_dir = Path(plot_dir)
    plot_dir.mkdir(exist_ok=True, parents=True)

    plot_image_grid(weights.mean(axis=1), f"{model_label} mean channel weights", True, get_cmap('PuOr'))
    plt.savefig(plot_dir / f"{model_label}_channels.png", bbox_inches='tight')
    plot_image_grid(weights.mean(axis=0), f"{model_label} mean plane weights", True, get_cmap('PiYG'))
    plt.savefig(plot_dir / f"{model_label}_planes.png", bbox_inches='tight')
    plt.show()


# base path that contains multiple models
base_path = "/home/jannis/Research/PommerData-Mirror/ml2/2022_08_12-20_02_10_sl_init_g99_m00_2e/"
# get all model names from that directory
model_names = list(
    map(
        lambda p: p.name,
        filter(
            lambda p: p.is_dir() and "model" in p.name,
            Path(base_path).iterdir()
        )
    )
)
# get the batch size 1 model paths
model_paths = [f"{base_path}/{name}/onnx/model-bsize-1.onnx" for name in model_names]

# plot the weights
for path, name in zip(model_paths, model_names):
    plot_first_layer(path, name, Path(base_path) / "plots")
