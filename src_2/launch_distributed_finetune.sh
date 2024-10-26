#!/bin/bash
# source ./hyperparams/params_pile.sh
export WORLD_SIZE=1
export NUM_GPU=8
# export ACC_STEPS=1
# export LOCAL_SIZE=1
data_path=$1
model_path=$2
output_path=$3
export TIMESTAMP=$( date +%Y-%m-%d_%H-%M-%S )
export MASTER_ADDR=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | head -n 1)
# export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
# export MASTER_PORT=54967
# export NEW_EXP_NAME=${TIMESTAMP}_${DATA_NAME}_${DATA_PATH}_${MODEL}_BSZ${BSZ}_ACC${ACC_STEPS}_BLOCKSZ${BLOCKSZ}_ATT${ATT_FUNC}_HLENGTH${H_LENGTH}_SAMPLETOPK${SAMPLETOPK}_MOEPDROP${MOE_PDROP}_GATING${GATING_SIZE}_TOPK${TOPK}_AUX${AUX_LOSS_TYPE}${AUX_LOSS_WEIGHT}
# export EXP_NAME=${1:-$NEW_EXP_NAME}
# export EXP_NAME=2023-03-16_00-23-23_all_the_pile_st-deep-16e_BSZ64_ACC1_BLOCKSZ1024_ATTstickbreaking_HLENGTH1024_SAMPLETOPK0_MOEPDROP0_GATING256_TOPK1_AUXmi0.001
# export OUTPUT_DIR=checkpoints/$EXP_NAME
# export LOG_DIR=$OUTPUT_DIR/logs
export OMP_NUM_THREADS=8

# mkdir -p ${OUTPUT_DIR}
# mkdir -p ${LOG_DIR}
echo 'here!'
bsub -q alt_6h -K -M 512G -gpu "num=$NUM_GPU/task:mode=exclusive_process" -n $WORLD_SIZE  \
	-R "select[infiniband  && h100,a100_80gb] rusage[mem=512G]" \
	-o train-distributed-${TIMESTAMP}.log \
	blaunch.sh bash -c "set -m; WORLD_SIZE=$WORLD_SIZE TIMESTAMP=$TIMESTAMP NUM_GPU=$NUM_GPU ./distribute_finetune.sh $data_path $model_path $output_path; wait" 
