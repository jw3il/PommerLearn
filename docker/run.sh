#!/bin/bash

if [[ -z "${POMMER_DATA_DIR}" ]]; then
    echo "Please set environment variable \$POMMER_DATA_DIR before calling this script."
    exit 1
fi

if [[ -z "${POMMER_SHM_SIZE}" ]]; then
    POMMER_SHM_SIZE=32g
    echo "Environment variable \$POMMER_SHM_SIZE is not set. Using default ${POMMER_SHM_SIZE}"
fi

if [[ -z "${TAG}" ]]; then
  TAG=tensorrt
fi

# Run the image and pass additional args with "$@"
docker run --gpus all -v "${POMMER_DATA_DIR}":/data --rm --shm-size="${POMMER_SHM_SIZE}" "pommer:${TAG}" \
    conda run --no-capture-output -n pommer \
    python -u /PommerLearn/pommerlearn/training/rl_loop.py --dir /data --exec /PommerLearn/build/PommerLearn "$@"
