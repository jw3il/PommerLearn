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
RUN wget -nv https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3.sh && \
    bash miniconda3.sh -b -p /opt/miniconda3 && \
    rm miniconda3.sh && \
    conda create -n pommer python=3.7 && \
    source /opt/miniconda3/bin/activate pommer && \
    # required for training
    conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch && \
    # required for cpp build
    conda install -y z5py xtensor boost nlohmann_json blosc catch2 -c conda-forge && \
    # main runtime dependencies
    pip install --no-input --no-cache-dir numpy matplotlib imageio zarr numcodecs tqdm sklearn tensorboard && \
    conda clean --all --yes


# set environment variables from parent image
ENV TENSORRT_PATH="/opt/tensorrt"
ENV CUDA_PATH="/usr/local/cuda"

# clone repository
WORKDIR /
RUN git clone https://gitlab.com/jweil/PommerLearn.git


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
    git submodule update

# build the binary
WORKDIR /PommerLearn/build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DUSE_TENSORRT8=ON -DCMAKE_CXX_COMPILER="$(which g++)" .. && \
    make VERBOSE=1 all -j$(nproc) && \
    # clear the cache
    rm -r CMakeFiles

# start environment
CMD ["conda", "run", "--no-capture-output", "-n", "pommer", "/bin/bash"]
