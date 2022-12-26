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

if [ -z "${POMMER_1VS1}" ] || [ "${POMMER_1VS1}" != true ]; then
    EXEC_PATH=/PommerLearn/build/PommerLearn
else
    EXEC_PATH=/PommerLearn/build1vs1/PommerLearn
    echo -e "\033[41;37m###########################################################\033[0m"
    echo -e "\033[41;37m#### Warning: running PommerLearn with the 1vs1 build. ####\033[0m"
    echo -e "\033[41;37m###########################################################\033[0m"
fi

# select entry point and pass additional args with "$@"
if [ -z "${MODE}" ] || [ "${MODE}" = "train" ]; then
    DOCKER_OPT=""
    CMD="conda run --no-capture-output -n pommer python -u /PommerLearn/pommerlearn/training/rl_loop.py --dir /data --exec ${EXEC_PATH} $@"
elif [ "${MODE}" = "exec" ]; then
    DOCKER_OPT=""
    CMD="${EXEC_PATH} $@"
elif [ "${MODE}" = "it" ]; then
    DOCKER_OPT="-it"
    CMD=""
else
    echo "Unknown mode '${MODE}'."
    exit 1;
fi

# Run the image
docker run --gpus all -v "${POMMER_DATA_DIR}":/data --rm --shm-size="${POMMER_SHM_SIZE}" ${DOCKER_OPT} "pommer:${TAG}" \
    $CMD
