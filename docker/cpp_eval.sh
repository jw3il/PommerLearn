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

# best models
declare -A MODEL_NAME_PATH_PAIRS
MODEL_NAME_PATH_PAIRS[sl]="/data/model-selected-sl/onnx"
MODEL_NAME_PATH_PAIRS[sl2rl]="/data/model-selected-sl2rl/onnx"
MODEL_NAME_PATH_PAIRS[rl]="/data/model-selected-rl/onnx"

# just one value allowed here
GAMES=1000
SIMULATIONS="1000"
TERMINAL_SOLVER="false"

# logfile path on host machine
LOGFILE="${POMMER_DATA_DIR}/$(date +%Y%m%d_%H%M%S)_eval_log.csv"

# create logfile and write column headers
echo "OpponentModel,SearchMode,TerminalSolver,ModelName,ModelPath,Simulations,Episodes,TotalSteps,Wins0,Alive0,Wins1,Alive1,Wins2,Alive2,Wins3,Alive3,Draws,NotDone,Time,EvalInfo" > "${LOGFILE}"

echo_command_string () {
    MODEL_NAME="$1"
    MODEL_PATH="${MODEL_NAME_PATH_PAIRS[${MODEL_NAME}]}"
    SEARCH_MODE="$2"
    OPPONENT_MODEL="$3"

    if [[ "$SIMULATIONS" = "0" ]]; then
        SIM_PARAM="--raw-net-agent"
    else
        SIM_PARAM="--simulations=${SIMULATIONS}"
    fi

    if [[ "$SEARCH_MODE" = "OnePlayer" ]]; then
        EXEC="POMMER_1VS1=false MODE=exec bash run.sh"
    else
        EXEC="POMMER_1VS1=true MODE=exec bash run.sh"
    fi

    CMD="$EXEC --gpu \$CUDA_VISIBLE_DEVICES --mode=ffa_mcts --model=${MODEL_PATH} ${SIM_PARAM} --planning-agents=${OPPONENT_MODEL} --max-games=${GAMES} --mctsSolver=${TERMINAL_SOLVER}"
    # get stats from last line and append stats with run configuration to logfile
    echo "$CMD | tail -n 1 | xargs -d \"\n\" printf \"${OPPONENT_MODEL},${SEARCH_MODE},${TERMINAL_SOLVER},${MODEL_NAME},${MODEL_PATH},${SIMULATIONS},%s\n\" >> ${LOGFILE}";
}

# run experiments
time (
(
    echo_command_string "sl" "OnePlayer" "SimpleUnbiasedAgent"
    echo_command_string "sl" "OnePlayer" "RawNetAgent"
    echo_command_string "sl" "TwoPlayer" "SimpleUnbiasedAgent"
    echo_command_string "sl" "TwoPlayer" "RawNetAgent"
    echo_command_string "sl2rl" "OnePlayer" "SimpleUnbiasedAgent"
    echo_command_string "sl2rl" "OnePlayer" "RawNetAgent"
    echo_command_string "rl" "OnePlayer" "SimpleUnbiasedAgent"
    echo_command_string "rl" "OnePlayer" "RawNetAgent"
) #| simple_gpu_scheduler --gpus $GPUS
)
