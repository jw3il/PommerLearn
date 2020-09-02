# Pommer-Learn: Learning-based MCTS in the Pommerman Environment

Idea: Combine Multi-Agent Reinforcement Learning and Monte-Carlo Tree Search (MCTS).

## Prerequisites

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

## Setup

### Download

This repository depends on submodules. Clone it recursively with

```
git clone --recurse-submodules git@gitlab.com:jweil/PommerLearn.git
```

### Build and Installation

* Build the C++ environment with the provided `CMakeLists.txt`.

    The current version requires you to set the env variable CONDA_ENV_PATH to the path of your conda environment (e.g. `~/conda/envs/pommer`)

* The python dependencies can be installed with

    ```
    pip install -r requirements.txt
    ```
