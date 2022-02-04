# Pommer-Learn: Learning-based MCTS in the Pommerman Environment

Idea: Combine Multi-Agent Reinforcement Learning and Monte-Carlo Tree Search (MCTS).

## Docker

The simplest way to get started and execute runs is to build a docker image and run it as a container
(currently only with TensorRT)

1. Clone the repository (optional, you can also just download the dockerfile)
2. Build the image
    ```
    docker build --build-arg BUILD_TIMESTAMMP=$(date +%Y%m%d-%H%M%S) -t pommer-tensorrt docker/tensorrt/
    ```
    The build arg is optional and accelerates rebuilding by invalidating only the code pull & build layers.

3. You can now run your image as a container and mount a data directory (`$POMMER_DATA_DIR`) to log the results

    a) Run in interactive mode
    ```
    docker run --gpus device=0 -v $POMMER_DATA_DIR:/data --rm -it pommer-tensorrt
    ```

    b) Directly start training
    ```
    docker run --gpus device=0 -v $POMMER_DATA_DIR:/data --rm pommer-tensorrt \
        conda run --no-capture-output -n pommer \
        python -u /PommerLearn/pommerlearn/training/rl_loop.py --dir /data --exec /PommerLearn/build/PommerLearn
    ```

    **Warning**: If you use rootless docker, the above commands will probably run out of memory. 
    Adding `--ipc=host` or `--shm-size=32g` to the `docker run` command helps. 

If there are changes in the repository, you can rebuild the docker image with `--no-cache`.

## Development

### Prerequisites

For the python side:

* `python 3.7`

    It is recommend to use virtual environments. This guide will use [Anaconda](https://www.anaconda.com/). Create an environment named `pommer` with

    ```
    conda create -n pommer python=3.7
    ```

* `pip`

For the C++ side:

* `gcc`

* `make`

* [z5](https://github.com/constantinpape/z5), [xtensor](https://github.com/xtensor-stack/xtensor), [boost](boost.org), [json by nlohmann](https://github.com/nlohmann/json/), [catch2](https://github.com/catchorg/Catch2)

    You can directly install these libraries with conda in the pommer env:

    ```
    conda install -c conda-forge z5py xtensor boost nlohmann_json blosc catch2
    ```

* [blaze](https://bitbucket.org/blaze-lib/blaze/src/master/)

    Blaze needs to be installed manually.

    * https://bitbucket.org/blaze-lib/blaze/downloads/

    * https://bitbucket.org/blaze-lib/blaze/wiki/Configuration%20and%20Installation#!manual-installation-on-linuxmacos

    ```
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/
    sudo make install
    export BLAZE_PATH=/usr/local/include/
    ```

(Tested on Ubuntu 20.04 LTS)

### Setup

#### Download

This repository depends on submodules. Clone it and initialize all submodules with

```
git clone git@gitlab.com:jweil/PommerLearn.git && \
cd PommerLearn && \
git submodule update --init deps/playground && \
git submodule update --init deps/pomcpp && \
git submodule update --init deps/CrazyAra
```

#### Build and Installation

* Build the C++ environment with the provided `CMakeLists.txt`.

    The current version requires you to set the env variables

    * `CONDA_ENV_PATH`: path of your conda environment (e.g. `~/conda/envs/pommer`)
    * `BLAZE_PATH`: blaze installation path (e.g. `/usr/local/include`)
    * `CUDA_PATH`: cuda installation path (e.g. `/usr/local/cuda`)
    * `TENSORRT_PATH` (when using the CrazyAra TensorRT backend, e.g. `/usr/src/tensorrt`)
    * `Torch_DIR` (when using the CrazyAra Torch backend)

* The python dependencies can be installed with

    ```
    pip install -r requirements.txt
    ```

#### Troubleshooting Notes
* In older versions of TensorRT, you have to manually comment out `using namespace sample;` in `deps/CrazyAra/engine/src/nn/tensorrtapi.cpp`
* Make sure that you've pulled all submodules recursively (including Stockfish for CrazyAra)

## Performance Profiling

Install the plotting utility for [gprof](https://ftp.gnu.org/old-gnu/Manuals/gprof-2.9.1/html_mono/gprof.html):
* https://github.com/jrfonseca/gprof2dot

Activate the CMake option `USE_PROFILING` in `CMakeLists.txt` and rebuild.
Run the executable and generate the plot:
```bash
./PommerLearn --mode ffa_mcts --max_games 10
gprof PommerLearn | gprof2dot | dot -Tpng -o profile.png
```
