# Pommer-Learn: Learning-based MCTS in the Pommerman Environment

Idea: Combine Multi-Agent Reinforcement Learning and Monte-Carlo Tree Search (MCTS).

## Prerequisites

For the python side:
* `python 3.7` (it is recommend to use virtual environments, e.g. with [Anaconda](https://www.anaconda.com/))
* `pip`

For the C++ side:
 * `gcc`
 * `make`
 
(Tested on Ubuntu 20.04 LTS)

## Setup

### Download

This repository depends on submodules. Clone it recursively with

```
git clone --recurse-submodules git@gitlab.com:jweil/PommerLearn.git
```

### Build and Installation

First, you have to build the C++ components and corresponding dependencies with

```
bash build.sh
```

The python dependencies can be installed with

```
pip install -r requirements.txt
```