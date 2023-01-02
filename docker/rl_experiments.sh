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
GPUS="0,1,2,3"

DATE=$(date +%Y%m%d_%H%M%S)

(
for model in "dummy" "sl"; do
cat << EndOfMessage
bash run.sh --gpu \$CUDA_VISIBLE_DEVICES --name-initials=JW --model="/data/model-${model}/" --comment="${model}_default" --it 200 > "${DATE}_${model}_default.log" 2>&1
bash run.sh --gpu \$CUDA_VISIBLE_DEVICES --name-initials=JW --model="/data/model-${model}/" --comment="${model}_default_1e" --it 200 --exec-args="--env-gen-seed-eps=1" > "${DATE}_${model}_default_1e.log" 2>&1
bash run.sh --gpu \$CUDA_VISIBLE_DEVICES --name-initials=JW --model="/data/model-${model}/" --comment="${model}_raw_net_opponent" --it 200 --exec-args="--1st-opponent-type=RawNetAgent" > "${DATE}_${model}_raw_net_opponent.log" 2>&1
bash run.sh --gpu \$CUDA_VISIBLE_DEVICES --name-initials=JW --model="/data/model-${model}/" --comment="${model}_raw_net_opponent_and_planning" --it 200 --exec-args="--1st-opponent-type=RawNetAgent --planning-agents=RawNetAgent" > "${DATE}_${model}_raw_net_opponent_and_planning.log" 2>&1
bash run.sh --gpu \$CUDA_VISIBLE_DEVICES --name-initials=JW --model="/data/model-${model}/" --comment="${model}_raw_net_opponent_and_planning_sd_2" --it 200 --exec-args="--1st-opponent-type=RawNetAgent --planning-agents=RawNetAgent --switch-depth=2" > "${DATE}_${model}_raw_net_opponent_and_planning_sd_2.log" 2>&1
EndOfMessage
done
) | simple_gpu_scheduler --gpus $GPUS
