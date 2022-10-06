# Pommer-Learn: Learning-based MCTS in the Pommerman Environment

Idea: Combine Multi-Agent Reinforcement Learning and Monte-Carlo Tree Search (MCTS).

## Docker

The simplest way to get started and execute runs is to build a docker image and run it as a container.

Available backends:
- TensorRT (**NVIDIA GPU required**): Tested with TensorRT 8.0.1 and PyTorch 1.9.0.

### Prerequisites

To use NVIDIA GPUs in docker containers, you have to install docker and nvidia-docker2. Have a look at the installation guide https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html.

### Build Scripts

We provide small scripts to facilitate building the image and running experiments.

1. Build the image
    ```
    $ bash docker/build.sh
    ```
    This automatically caches the dependencies. If you run it again, only the code is rebuilt. If you want to rebuild the whole image, just call `bash docker/build.sh --no-cache`.

1. Specify where you want to store the data generated by the experiments as environment variable `$POMMER_DATA_DIR`. You can `export POMMER_DATA_DIR=/some/dir` or just add `POMMER_DATA_DIR=/some/dir` as a prefix to the command in the following step.

3. Create a container and run the training loop (replace `--help` with the arguments of your choice)
    ```
    $ bash docker/run.sh --help
    ```
    * Note that `--dir` and `--exec` are already specified correctly by `docker/run.sh`.
    * All GPUs are visible in the container and gpu 0 is used by default. You can specify the gpu to be used like `--gpu 4`.

### Manual Docker Build

Of course, you can also build and run the image manually. Have a closer look at the scripts from the previous section for details.

Additional notes:
* You can limit the gpu access of a container like `--gpus device=4`. However, PommerLearn has a `--gpu` argument that can be used instead.
* **Warning**: If you use rootless docker, the container will probably run out of memory. 
    Adding `--ipc=host` or `--shm-size=32g` to the `docker run` command helps.
    This is also done by default in `docker/run.sh`.

## Development

### Manual Installation of Dependencies

For the python side:

* `python 3.7` and `pip`

    It is recommend to use virtual environments. This guide will use [Anaconda](https://www.anaconda.com/). Create an environment named `pommer` with

    ```
    $ conda create -n pommer python=3.7
    ```

For the C++ side:

* Essential build tools: `gcc`, `make`, `cmake`

    ```
    $ sudo apt install build-essential cmake
    ```

* The dependencies [z5](https://github.com/constantinpape/z5), [xtensor](https://github.com/xtensor-stack/xtensor), [boost](boost.org) and [json by nlohmann](https://github.com/nlohmann/json/) can directly be installed with conda in the pommer environment:

    ```
    (pommer) $ conda install -c conda-forge z5py xtensor boost nlohmann_json blosc
    ```

* [Blaze](https://bitbucket.org/blaze-lib/blaze/src/master/) needs to be installed manually. Note that it can be unpacked anywhere, it does not have to be `/usr/local`. For further information, you can refer to the [installation guide](https://bitbucket.org/blaze-lib/blaze/wiki/Configuration%20and%20Installation#!manual-installation-on-linuxmacos) or the Dockerfiles in this repository.

    ```
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/
    sudo make install
    export BLAZE_PATH=/usr/local/include/
    ```
* Manual installation of **TensorRT** (not Torch-TensorRT), including CUDA and cuDNN. Please refer to the installation guide by NVIDIA https://developer.nvidia.com/tensorrt-getting-started.

### Clone Repository

This repository depends on submodules. Clone it and initialize all submodules with

```
$ git clone git@gitlab.com:jweil/PommerLearn.git && \
$ cd PommerLearn && \
$ git submodule update --init
```

### How to build

1. The current version requires you to set the env variables

    * `CONDA_ENV_PATH`: path of your conda environment (e.g. `~/conda/envs/pommer`)
    * `BLAZE_PATH`: blaze installation path (e.g. `/usr/local/include`)
    * `CUDA_PATH`: cuda installation path (e.g. `/usr/local/cuda`)
    * `TENSORRT_PATH` (when using the CrazyAra TensorRT backend, e.g. `/usr/src/tensorrt`)
    * [`Torch_DIR`] (when using the CrazyAra Torch backend, currently untested)

2. Build the C++ environment with the provided `CMakeLists.txt`. To use TensorRT >= 8 (recommended), you have to specify `-DUSE_TENSORRT8=ON`.

```
/PommerLearn/build $ cmake -DCMAKE_BUILD_TYPE=Release -DUSE_TENSORRT8=ON -DCMAKE_CXX_COMPILER="$(which g++)" ..
/PommerLearn/build $ make VERBOSE=1 all -j8
```

### How to run

Optional: You can install PyTorch 1.9.0 with GPU support via

```
conda install -y pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c conda-forge -c pytorch
```

The remaining python runtime dependencies can be installed with
```
(pommer) $ pip install -r requirements.txt
```

Before starting the RL loop, you can check whether everything is set up correctly by creating a dummy model and loading it in the cpp executable:

```
(pommer) /PommerLearn/build $ python ../pommerlearn/debug/create_dummy_model.py
(pommer) /PommerLearn/build $ ./PommerLearn --mode=ffa_mcts --model=./model/onnx
```

You can then start training by running

```
(pommer) /PommerLearn/build $ python ../pommerlearn/training/rl_loop.py
```

### Troubleshooting

Prerequisites and Building
* Make sure that you've pulled all submodules recursively
* In older versions of TensorRT, you have to manually comment out `using namespace sample;` in `deps/CrazyAra/engine/src/nn/tensorrtapi.cpp`
* We experienced issues with `std::filesystem` being undefined when using GCC 7.5.0. We recommend to update to more recent versions, e.g. GCC 11.2.0.

Running
* For runtime issues like `libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`, try loading your system libraries with `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/`. 
  On some systems, ctypes somehow uses a different libstdc++ from the conda environment instead of the correct lib path. 
  As a last resort, you can back up the original library `mv /conda-lib-path/libstdc++.so.6 /conda-lib-path/libstdc++.so.6.old` and then create a symbolic link `ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /conda-lib-path/libstdc++.so.6`.
* If you encounter errors like `ModuleNotFoundError: No module named 'training'`, set your `PYTHONPATH` to the `pommerlearn` directory. For example, `export PYTHONPATH=/PommerLearn/pommerlearn`.
* When loading `tensorboard` runs, you can get errors like `Error: tonic::transport::Error(Transport, hyper::Error(Accept, Os { code: 24, kind: Other, message: "Too many open files" }))`. The argument `--load_fast=false` might help.

### Performance Profiling

Install the plotting utility for [gprof](https://ftp.gnu.org/old-gnu/Manuals/gprof-2.9.1/html_mono/gprof.html):
* https://github.com/jrfonseca/gprof2dot

Activate the CMake option `USE_PROFILING` in `CMakeLists.txt` and rebuild.
Run the executable and generate the plot:
```bash
./PommerLearn --mode ffa_mcts --max_games 10
gprof PommerLearn | gprof2dot | dot -Tpng -o profile.png
```
