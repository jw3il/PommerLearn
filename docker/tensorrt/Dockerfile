# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/tensorrt:21.07-py3

# get blaze and install its dependencies + libx11-dev as runtime requirement
ENV BLAZE_PATH="/usr/local/include/blaze"
RUN apt-get update && apt-get -y install libblas-dev liblapack-dev libx11-dev && apt-get -y clean && \
    wget -nv https://bitbucket.org/blaze-lib/blaze/downloads/blaze-3.8.tar.gz && \
    tar -xf blaze-3.8.tar.gz && \
    cp -r ./blaze-3.8/blaze /usr/local/include && \
    rm -r blaze-3.8 && \
    rm blaze-3.8.tar.gz

# install conda dependencies
ENV PATH="/opt/miniconda3/bin:${PATH}"
ENV CONDA_ENV_PATH="/opt/miniconda3/envs/pommer"
RUN wget -nv https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh -O miniconda3.sh && \
    bash miniconda3.sh -b -p /opt/miniconda3 && \
    rm miniconda3.sh && \
    conda create -n pommer python=3.7 && \
    source /opt/miniconda3/bin/activate pommer && \
    # required for cpp build
    conda install -y -c conda-forge z5py xtensor boost nlohmann_json blosc catch2 && \
    # main runtime dependencies
    pip install --no-input --no-cache-dir numpy matplotlib imageio torch zarr numcodecs tqdm sklearn tensorboard && \
    conda clean --all --yes
    
# set environment variables from parent image
ENV TENSORRT_PATH="/opt/tensorrt"
ENV CUDA_PATH="/usr/local/cuda"

# clone repository 
WORKDIR /
RUN git clone --recurse-submodules https://gitlab.com/jweil/PommerLearn.git

# build the binary
WORKDIR /PommerLearn/build
RUN cmake -DUSE_TENSORRT8=ON -DCMAKE_CXX_COMPILER="$(which g++)" .. && \
    make VERBOSE=1 all -j 8 && \
    # clear the cache
    rm -r CMakeFiles

# install python runtime dependencies
WORKDIR /PommerLearn
ENV PYTHONPATH=/PommerLearn/pommerlearn
RUN source /opt/miniconda3/bin/activate pommer && \
    pip install -r requirements.txt

# start environment
CMD ["conda", "run", "--no-capture-output", "-n", "pommer", "/bin/bash"]

