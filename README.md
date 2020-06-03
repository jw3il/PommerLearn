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

* [z5](https://github.com/constantinpape/z5), [xtensor](https://github.com/xtensor-stack/xtensor), [boost](boost.org), [json by nlohmann](https://github.com/nlohmann/json/)

    You can directly install these libraries with conda in the pommer env:

    ```
    conda install -c conda-forge z5py xtensor boost nlohmann_json
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

    The current version requires you to set the env variable CONDA_INCLUDE_PATH to your conda include path (e.g. `~/conda/envs/pommer/include`)

* The python dependencies can be installed with

    ```
    pip install -r requirements.txt
    ```
