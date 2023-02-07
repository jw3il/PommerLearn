#!/bin/bash

if [[ -z "${POMMER_DATA_DIR}" ]]; then
    echo "Please set environment variable \$POMMER_DATA_DIR before calling this script."
    exit 1
fi

if [ ! -f "run.sh" ]; then
    echo "Please run this script in the docker directory, run.sh could not be found."
    exit 1
fi

# define gpus to be used as a list
# simply use gpu indices multiple times if you want to schedule
# multiple jobs per gpu 
GPUS="0,0,1,1,2,2,3,3"

# model paths within the docker containers
declare -a MODEL_NAME_PATH_PAIRS=(
    "sl-0 /data/model-sl-0/onnx"
    "sl-1 /data/model-sl-1/onnx"
    "sl-2 /data/model-sl-2/onnx"
    "sl-3 /data/model-sl-3/onnx"
    "sl-4 /data/model-sl-4/onnx"
    "dummy-0 /data/model-dummy-0/onnx"
    "dummy-1 /data/model-dummy-1/onnx"
    "dummy-2 /data/model-dummy-2/onnx"
    "dummy-3 /data/model-dummy-3/onnx"
    "dummy-4 /data/model-dummy-4/onnx"
)

GAMES=1000
SIMULATIONS="0 50 100 250 500 1000"
SEARCH_MODES="OnePlayer TwoPlayer"
TERMINAL_SOLVER="false"

# logfile path on host machine
LOGFILE="${POMMER_DATA_DIR}/$(date +%Y%m%d_%H%M%S)_pommer_log.csv"

# create logfile and write column headers
echo "SearchMode,TerminalSolver,ModelName,ModelPath,Simulations,Episodes,TotalSteps,Wins0,Alive0,Wins1,Alive1,Wins2,Alive2,Wins3,Alive3,Draws,NotDone,Time,EvalInfo" > "${LOGFILE}"

# TODO: Add terminal solver option

# run experiments
time (
for sim in ${SIMULATIONS}; do
for pair in "${MODEL_NAME_PATH_PAIRS[@]}"; do
for search in ${SEARCH_MODES}; do
for tsolver in ${TERMINAL_SOLVER}; do
    read -a strarr <<< "$pair"
    model_name=${strarr[0]}
    model_path=${strarr[1]}

    if [[ "$sim" = "0" ]]; then
        SIM_PARAM="--raw-net-agent"
    else
        SIM_PARAM="--simulations=${sim}"
    fi

    if [[ "$search" = "OnePlayer" ]]; then
        EXEC="POMMER_1VS1=false MODE=exec bash run.sh"
    else
        EXEC="POMMER_1VS1=true MODE=exec bash run.sh"
    fi

    # main command
    CMD="$EXEC --gpu \$CUDA_VISIBLE_DEVICES --mode=ffa_mcts --model=${model_path} ${SIM_PARAM} --max-games=${GAMES} --mctsSolver=${tsolver} --track-stats"
    # get stats from last line and append stats with run configuration to logfile
    echo "$CMD | tail -n 1 | xargs -d \"\n\" printf \"${search},${tsolver},${model_name},${model_path},${sim},%s\n\" >> ${LOGFILE}";
done
done
done
done | simple_gpu_scheduler --gpus $GPUS
)
