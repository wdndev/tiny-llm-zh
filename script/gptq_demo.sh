#!/bin/bash

set -x

# GPTQ parameters
BITS=4  # [4, 8]
GROUP_SIZE=128
DAMP_PERCENT=0.1
DESC_ACT=False
STATIC_GROUPS=False
SYM=True
TRUE_SEQUENTIAL=True

# training parameters
MAX_LEN=8192
BATCH_SIZE=1
CACHE_ON_GPU=False
USR_TRITON=False

# basic setting
MODEL_PATH="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/wangdongnian/outputs/ckpt/qwen2_7b_sft_v10_qwen2-20240724-173728/iter_0007484/huggingface_format"
OUTPUT_DIR="outputs"
DATASEY_PATH="quant_data/v10_new_prompt_data.jsonl"
N_GPUS=4
GPU_MAX_MEMORY=20
MODEL_NAME="qwen2_7b_v10_new_prompt"

OUTPUT_MODEL_PATH=${OUTPUT_DIR}/${MODEL_NAME}_gptq_int${BITS}
mkdir -p $OUTPUT_MODEL_PATH
QUANT_LOG="${OUTPUT_MODEL_PATH}/quantize_$(date "+%Y%m%d%H%M").log"

GPTQ_ARGS=" \
    --bits ${BITS} \
    --group_size ${GROUP_SIZE} \
    --damp_percent ${DAMP_PERCENT} \
    --desc_act ${DESC_ACT} \
    --static_groups ${STATIC_GROUPS} \
    --sym ${SYM} \
    --true_sequential ${TRUE_SEQUENTIAL} \
"

TRAIN_ARGS=" \
    --max_len ${MAX_LEN} \
    --batch_size ${BATCH_SIZE} \
    --cache_examples_on_gpu ${CACHE_ON_GPU} \
    --use_triton ${USR_TRITON} \
"

SCRIPT_ARGS=" \
    --model_id  ${MODEL_PATH} \
    --dataset_dir_or_path ${DATASEY_PATH} \
    --quant_output_dir ${OUTPUT_MODEL_PATH} \
    --ngpus ${N_GPUS} \
    --gpu_max_memory ${GPU_MAX_MEMORY} \
"

ALL_ARGS=" $GPTQ_ARGS $TRAIN_ARGS $SCRIPT_ARGS "

LAUNCHER="python quantize/gptq_quantize.py "

# Combine all arguments into one command
CMD="$LAUNCHER $ALL_ARGS"

# Print the command that will be executed for debugging purposes
echo $CMD

# Execute the quantization process and redirect all output to the log file
nohup $CMD > ${QUANT_LOG} 2>&1 &

# Notify the user about the location of the log file
echo "Running successfully. The logs are saved in ${QUANT_LOG}"