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
    "sl /data/model-sl/onnx"
    "dummy /data/model-dummy/onnx"
)

GAMES=1000
SIMULATIONS="0 50 100 250 500 1000"
SEARCH_MODES="OnePlayer TwoPlayer"
TERMINAL_SOLVER="true false"

# logfile path on host machine
LOGFILE="${POMMER_DATA_DIR}/$(date +%Y%m%d_%H%M%S)_pommer_log.csv"

# create logfile and write column headers
echo "SearchMode,TerminalSolver,ModelName,ModelPath,Simulations,Episodes,TotalSteps,Wins0,Alive0,Wins1,Alive1,Wins2,Alive2,Wins3,Alive3,Draws,NotDone,Time,EvalInfo" > "${LOGFILE}"

# TODO: Add terminal solver option

# run experiments
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
    CMD="$EXEC --gpu \$CUDA_VISIBLE_DEVICES --mode=ffa_mcts --model=${model_path} ${SIM_PARAM} --max-games=${GAMES} --mctsSolver=${tsolver}"
    # get stats from last line and append stats with run configuration to logfile
    echo "$CMD | tail -n 1 | xargs printf \"${search},${tsolver},${model_name},${model_path},${sim},%s\n\" >> ${LOGFILE}";
done
done
done
done | simple_gpu_scheduler --gpus $GPUS
