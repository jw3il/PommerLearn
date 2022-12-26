#!/bin/bash

# define gpus to be used as a list
# simply use gpu indices multiple times if you want to schedule
# multiple jobs per gpu 
GPUS="0,0"

# TODO: use docker image instead
EXEC="./PommerLearn"
MODEL_DIR="./"

declare -a MODEL_NAME_PATH_PAIRS=(
    "trained ${MODEL_DIR}/2022_12_14-16_06_21_from_zero_new_planes_e1_i200_199_model/onnx"
    "initial ${MODEL_DIR}/dummy-model/onnx"
)

GAMES=10 #100
SIMULATIONS="0 25" #"0 50 100 250 500 1000"

LOGFILE="$(date +%Y%m%d_%H%M%S)_pommer_log.csv"

# write headers to logfile
echo "ModelName,ModelPath,Simulations,Episodes,TotalSteps,Wins0,Alive0,Wins1,Alive1,Wins2,Alive2,Wins3,Alive3,Draws,NotDone,Time" > ${LOGFILE}

# TODO: Add terminal solver, single-player and two-player search options

# run experiments
for sim in ${SIMULATIONS}; do
    for pair in "${MODEL_NAME_PATH_PAIRS[@]}"; do
        read -a strarr <<< "$pair"
        model_name=${strarr[0]}
        model_path=${strarr[1]}

        if [ $sim = "0" ]
        then
            SIM_PARAM="--raw-net-agent"
        else
            SIM_PARAM="--simulations=${sim}"
        fi

        # main command
        CMD="$EXEC --gpu \$CUDA_VISIBLE_DEVICES --mode=ffa_mcts --model=${model_path} ${SIM_PARAM} --max-games=${GAMES}"
        # get stats from last line and append stats with run configuration to logfile
        echo "$CMD | tail -n 1 | xargs printf \"${model_name},${model_path},${sim},%s\n\" >> ${LOGFILE}"; 
    done
done | simple_gpu_scheduler --gpus $GPUS
