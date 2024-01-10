# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/tensorrt:21.07-py3

# install dependencies and get blaze
ENV BLAZE_PATH="/usr/local/include/blaze"
RUN apt-get update && \
    # general dependencies
    apt-get -y install git wget libx11-dev \
    # blaze dependencies
    libblas-dev liblapack-dev && \
    apt-get -y clean && \
    # get and unpack blaze
    wget -nv https://bitbucket.org/blaze-lib/blaze/downloads/blaze-3.8.tar.gz && \
    tar -xf blaze-3.8.tar.gz && \
    cp -r ./blaze-3.8/blaze /usr/local/include && \
    rm -r blaze-3.8 && \
    rm blaze-3.8.tar.gz

# install conda dependencies
ENV PATH="/opt/miniconda3/bin:${PATH}"
ENV CONDA_ENV_PATH="/opt/miniconda3/envs/pommer"
RUN wget -nv https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3.sh && \
    bash miniconda3.sh -b -p /opt/miniconda3 && \
    rm miniconda3.sh && \
    # the latest release does not always contain the latest version
    conda update -n base -c defaults conda && \
    conda create -n pommer python=3.7 && \
    source /opt/miniconda3/bin/activate pommer && \
    # required for training
    pip3 install --no-input torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    # required for cpp build
    conda install -y z5py xtensor boost nlohmann_json blosc -c conda-forge && \
    # main runtime dependencies
    pip install --no-input --no-cache-dir numpy matplotlib imageio zarr numcodecs tqdm scikit-learn tensorboard && \
    conda clean --all --yes

# set environment variables from parent image
ENV TENSORRT_PATH="/opt/tensorrt"
ENV CUDA_PATH="/usr/local/cuda"

# clone repository
WORKDIR /
RUN git clone https://github.com/jw3il/PommerLearn.git

# initial submodule checkout (not recursive) + install python runtime dependencies
WORKDIR /PommerLearn
ENV PYTHONPATH=/PommerLearn/pommerlearn
RUN git submodule update --init && \
    source /opt/miniconda3/bin/activate pommer && \
    pip install --no-cache-dir -r requirements.txt

# trick from https://stackoverflow.com/questions/35154219/rebuild-docker-image-from-specific-step
# use 
#   --build-arg BUILD_TIMESTAMMP=$(date +%Y%m%d-%H%M%S)
# to invalidate the cache and only rebuild the important part.
# note that you have to invalidate the cache if the requirements change.
ARG BUILD_TIMESTAMMP=unknown

# update repository
WORKDIR /PommerLearn
RUN git pull && \
    # optional: checkout specific branch
    # git checkout master && \
    git submodule update

# build the binary
WORKDIR /PommerLearn/build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DUSE_TENSORRT8=ON -DCMAKE_CXX_COMPILER="$(which g++)" .. && \
    make VERBOSE=1 all -j8 && \
    # clear the cache
    rm -r CMakeFiles

WORKDIR /PommerLearn/build1vs1
RUN cmake -DCMAKE_BUILD_TYPE=Release -DUSE_TENSORRT8=ON -DUSE_1VS1_MCTS=ON -DCMAKE_CXX_COMPILER="$(which g++)" .. && \
    make VERBOSE=1 all -j8 && \
    # clear the cache
    rm -r CMakeFiles

# start environment
CMD ["conda", "run", "--no-capture-output", "-n", "pommer", "/bin/bash"]
