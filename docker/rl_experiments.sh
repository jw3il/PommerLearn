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

DATE=$(date +%Y%m%d_%H%M%S)
DIRNAME="${DATE}_rl"
SIMULATIONS=250

mkdir -p $DIRNAME

(
for twoplayersearch in "true" "false"; do
for model in "dummy" "sl"; do
for rep in {0..4}; do
RUN_NAME="${model}_${SIMULATIONS}s_2${twoplayersearch}_exploit_${rep}"
cat << EndOfMessage
POMMER_1VS1=${twoplayersearch} bash run.sh --gpu \$CUDA_VISIBLE_DEVICES --model="/data/model-${model}-${rep}/" --name-initials=JW --comment="${RUN_NAME}" --it 50 --exec-args="--mode=ffa_mcts --simulations=${SIMULATIONS} --mctsSolver=false --targeted-samples=100000 --chunk-count=101 --track-stats" --train-args="--discount_factor=1 --mcts_val_weight=None --policy_loss_argmax_target=0.5" --num-latest=4 --num-recent=0 > "${DIRNAME}/${RUN_NAME}.log" 2>&1
EndOfMessage
done
done
done
) | simple_gpu_scheduler --gpus "$GPUS"
