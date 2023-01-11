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
GPUS="0,0,1,1,2,2"

DATE=$(date +%Y%m%d_%H%M%S)
DIRNAME="${DATE}_rl"

mkdir -p $DIRNAME

(
for twoplayersearch in "true" "false"; do
for model in "dummy" "sl"; do
cat << EndOfMessage
POMMER_1VS1=${twoplayersearch} bash run.sh --gpu \$CUDA_VISIBLE_DEVICES --name-initials=JW --model="/data/model-${model}/" --comment="${model}_100s_2${twoplayersearch}_noterm" --it 100 --exec-args="--simulations=100 --use-terminal-solver=false" > "${DIRNAME}/${model}_100s_2${twoplayersearch}_noterm.log" 2>&1
POMMER_1VS1=${twoplayersearch} bash run.sh --gpu \$CUDA_VISIBLE_DEVICES --name-initials=JW --model="/data/model-${model}/" --comment="${model}_250s_2${twoplayersearch}_noterm" --it 100 --exec-args="--simulations=250 --use-terminal-solver=false" > "${DIRNAME}/${model}_250s_2${twoplayersearch}_noterm.log" 2>&1
POMMER_1VS1=${twoplayersearch} bash run.sh --gpu \$CUDA_VISIBLE_DEVICES --name-initials=JW --model="/data/model-${model}/" --comment="${model}_500s_2${twoplayersearch}_noterm" --it 100 --exec-args="--simulations=500 --use-terminal-solver=false" > "${DIRNAME}/${model}_500s_2${twoplayersearch}_noterm.log" 2>&1
EndOfMessage
done
done
) | simple_gpu_scheduler --gpus "$GPUS"
