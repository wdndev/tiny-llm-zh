#!/bin/bash

set -x

# export CUDA_VISIBLE_DEVICES="4,5,6,7"

source /home/.bashrc
source /home/miniconda3/etc/profile.d/conda.sh
conda activate md_llm
which python

function killall {                                                                                                                                                                                      
    echo `ps -ef | grep $1 | grep -v grep | awk '{print $2}'`
    ps -ef | grep $1 | grep -v grep | awk '{print $2}' |xargs kill -9
}

WORK_DIR="/wangdongnian/personal/tiny-llama2.zh"
cd ${WORK_DIR}


VERSION=-1  # [0, 1, 2, -1]
DTYPE=fp32  # [fp16, fp32]
MODEL_PATH="outputs/ckpt/ptm_tiny_llama2_9m_epoch3/last_model.pth"
EXPORT_TYPE=checkpoint # [checkpoint, hf, meta-llama]

# 输出配置
MODEL_SIZE="9m" # [9m, 24m, 58m, 134m, 268m]
MODEL_NAME="hf_tiny_llama2_${MODEL_SIZE}"
OUTPUT_DIR="outputs/hf_model/${MODEL_NAME}"
mkdir -p $OUTPUT_DIR

if [[ $MODEL_SIZE == "9m" ]];then
    MAX_SEQ_LEN=512
    DIM=120
    N_LAYERS=6
    N_HEADS=6
    N_KV_HEADS=6
    MULTIPLE_OF=32
    DROPOUT=0.0
    VOCAB_SIZE=64793
elif [[ $MODEL_SIZE == "24m" ]];then
    MAX_SEQ_LEN=512
    DIM=288
    N_LAYERS=6
    N_HEADS=6
    N_KV_HEADS=6
    MULTIPLE_OF=32
    DROPOUT=0.1
    VOCAB_SIZE=64793
elif [[ $MODEL_SIZE == "58m" ]];then
    MAX_SEQ_LEN=512
    DIM=512
    N_LAYERS=8
    N_HEADS=8
    N_KV_HEADS=8
    MULTIPLE_OF=32
    DROPOUT=0.1
    VOCAB_SIZE=64793
elif [[ $MODEL_SIZE == "134m" ]];then
    MAX_SEQ_LEN=1024
    DIM=768
    N_LAYERS=12
    N_HEADS=12
    N_KV_HEADS=12
    MULTIPLE_OF=32
    DROPOUT=0.1
    VOCAB_SIZE=64793
elif [[ $MODEL_SIZE == "268m" ]];then
    MAX_SEQ_LEN=1024
    DIM=1024
    N_LAYERS=16
    N_HEADS=16
    N_KV_HEADS=16
    MULTIPLE_OF=32
    DROPOUT=0.1
    VOCAB_SIZE=64793
fi

GPT_ARGS=" \
    --max_seq_len ${MAX_SEQ_LEN} \
    --dim ${DIM} \
    --n_layers ${N_LAYERS} \
    --n_heads ${N_HEADS} \
    --n_kv_heads ${N_KV_HEADS} \
    --multiple_of ${MULTIPLE_OF} \
    --dropout ${DROPOUT} \
    --vocab_size ${VOCAB_SIZE} \
    "

CONTROL_ARGS=" \
    --output_dir ${OUTPUT_DIR} \
    --version ${VERSION} \
    --dtype ${DTYPE} \
    "

if [ "$EXPORT_TYPE" == "checkpoint" ];then
    CONTROL_ARGS+=" \
        --checkpoint ${MODEL_PATH} \
        "
elif [ "$EXPORT_TYPE" == "hf" ];then
    CONTROL_ARGS+=" \
        --hf ${MODEL_PATH} \
        "
elif [ "$EXPORT_TYPE" == "meta-llama" ];then
    CONTROL_ARGS+=" \
        --meta-llama ${MODEL_PATH} \
        "
fi

# 所有参数
ALL_ARGS=" $GPT_ARGS $CONTROL_ARGS "

export CMD="python export.py $ALL_ARGS "
echo $CMD

# 导出
$CMD

echo "export successful : ${OUTPUT_DIR}"