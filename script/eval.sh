#!/bin/bash

# set -x

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


EVAL_TYPE="ptm" # ["ptm", "sft"]
MODEL_SIZE="58m" # [9m, 24m, 58m, 134m, 268m]
EXPORT_TYPE=checkpoint # [checkpoint, hf, meta-llama]
MODEL_PATH="outputs/ckpt/ptm_tiny_llama2_webnovel_58m_epoch3/last_model.pth"

# generate
MAX_NEW_TOKENS=100
TEMPERATURE=1.0
TOP_K=30

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

GENERATE_ARGS=" \
    --eval_type ${EVAL_TYPE}\
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE} \
    --top_k ${TOP_K} \
    "

if [ "$EXPORT_TYPE" == "checkpoint" ];then
    GENERATE_ARGS+=" \
        --checkpoint ${MODEL_PATH} \
        "
elif [ "$EXPORT_TYPE" == "hf" ];then
    GENERATE_ARGS+=" \
        --hf ${MODEL_PATH} \
        "
fi

# 所有参数
ALL_ARGS=" $GPT_ARGS $GENERATE_ARGS "

export CMD="python eval.py $ALL_ARGS "
echo $CMD

# 导出
$CMD

echo "Eval successful : ${OUTPUT_DIR}"