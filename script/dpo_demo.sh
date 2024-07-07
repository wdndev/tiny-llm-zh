#!/bin/bash

set -x

# export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"

source /home/.bashrc
source /home/miniconda3/etc/profile.d/conda.sh
conda activate md_llm
which python

function killall {                                                                                                                                                                                      
    echo `ps -ef | grep $1 | grep -v grep | awk '{print $2}'`
    ps -ef | grep $1 | grep -v grep | awk '{print $2}' |xargs kill -9
}

WORK_DIR="/personal/tiny-llm-zh"
cd ${WORK_DIR}

# 常见参数
N_NODES=1
N_GPUS=8
MBS=32 # 单卡bs
GAS=1 # 梯度累积
GRAD_CLIP=1     # 梯度裁剪
RANK=0
MASTER_ADDR=`hostname -i`
MASTER_PORT=9902

LR=3e-4 # 初始学习率
LR_SCHEDULER_TYPE="cosine"
WARMUP_RATION=0.03

TRAIN_EPOCHS=5          # 训练轮次
LOGGING_STEPS=100       # 记录日志步数
CKPT_SAVE_STEPS=15000    # ckpt保存步数

SEED=12
DS_DTYPE="fp16" # [fp16, bf16]
RESUME="False"

# 数据
MODE="rl" # [ptm, sft, rm, rl]
DATASET_DIR_OR_PATH="data/rl_train/rl_train_data.jsonl"
BASE_MODEL_PATH="outputs/ckpt/sft_tiny_llm_92m_epoch5/last_sft_model"

MODEL_SIZE="92m" # [16m, 42m, 92m, 210m, 440m]
MODEL_NAME="${MODE}_tiny_llm_${MODEL_SIZE}"
OUTPUT_DIR="outputs/ckpt/${MODEL_NAME}_epoch${TRAIN_EPOCHS}"
mkdir -p $OUTPUT_DIR
TRAIN_LOG="${OUTPUT_DIR}/train_$(date "+%Y%m%d%H%M").log"
# tensorboard输出路径
TB_DIR="outputs/tensorboard/${MODEL_NAME}_epoch${TRAIN_EPOCHS}"
mkdir -p $TB_DIR

TRAIN_ARGS=""

DS_CONFIG_JSON=${OUTPUT_DIR}/${MODEL_SIZE}_ds_config.json
ZERO_STAGE=2

if [ $DS_DTYPE = "fp16" ];then
    TRAIN_ARGS+=" \
        --fp16 \
        "
    DS_FP16=true
    DS_BF16=false
    GAS_DTYPE=$DS_DTYPE
elif [ $DS_DTYPE = "bf16" ];then
    TRAIN_ARGS+=" \
        --bf16 \
        "
    DS_FP16=false
    DS_BF16=true
    GAS_DTYPE="fp32"

fi

cat <<EOT > $DS_CONFIG_JSON
{
  "train_micro_batch_size_per_gpu": $MBS,
  "train_batch_size": "auto",
  "gradient_clipping": ${GRAD_CLIP},
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": ${DS_BF16}
  },
  "data_types": {
    "grad_accum_dtype": "${GAS_DTYPE}"
  },
  "fp16": {
    "enabled": ${DS_FP16},
    "loss_scale": 0,
    "loss_scale_window": 200,
    "hysteresis": 5,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "steps_per_print": 10,
  "wall_clock_breakdown": true,
  "comms_logger": {
      "enabled": true,
      "verbose": false,
      "prof_all": false,
      "debug": false
    },
    "flops_profiler": {
        "enabled": false,
        "profile_step": 30,
        "module_depth": -1,
        "top_modules": 1,
        "detailed": true,
        "output_file": null
    }
}
EOT


TRAIN_ARGS+=" \
    --seed ${SEED} \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --deepspeed ${DS_CONFIG_JSON} \
    --per_device_train_batch_size ${MBS} \
    --gradient_accumulation_steps ${GAS} \
    --do_train \
    --num_train_epochs ${TRAIN_EPOCHS} \
    --logging_dir ${TB_DIR} \
    --logging_strategy steps \
    --logging_steps ${LOGGING_STEPS} \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta1 0.95 \
    --max_grad_norm ${GRAD_CLIP} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
    --learning_rate ${LR} \
    --warmup_ratio ${WARMUP_RATION} \
    --weight_decay 0.01 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps ${CKPT_SAVE_STEPS} \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --save_safetensors False \
    --ddp_find_unused_parameters False \
"

if [[ $MODEL_SIZE == "16m" ]];then
    HIDDEN_SIZE=120
    NUM_HIDDEN_LAYERS=6
    NUM_ATTENTION_HEADS=6
    INTERMEDIATE_SIZE=384
    ROPE_THETA=10000.0
    MAX_POSITION_EMBEDDINGS=512
    VOCAB_SIZE=64798
elif [[ $MODEL_SIZE == "42m" ]];then
    HIDDEN_SIZE=288
    NUM_HIDDEN_LAYERS=6
    NUM_ATTENTION_HEADS=6
    INTERMEDIATE_SIZE=768
    ROPE_THETA=10000.0
    MAX_POSITION_EMBEDDINGS=512
    VOCAB_SIZE=64798
elif [[ $MODEL_SIZE == "92m" ]];then
    HIDDEN_SIZE=512
    NUM_HIDDEN_LAYERS=8
    NUM_ATTENTION_HEADS=8
    INTERMEDIATE_SIZE=1408
    ROPE_THETA=10000.0
    MAX_POSITION_EMBEDDINGS=1024
    VOCAB_SIZE=64798
elif [[ $MODEL_SIZE == "210m" ]];then
    HIDDEN_SIZE=768
    NUM_HIDDEN_LAYERS=16
    NUM_ATTENTION_HEADS=12
    INTERMEDIATE_SIZE=2048
    ROPE_THETA=10000.0
    MAX_POSITION_EMBEDDINGS=1024
    VOCAB_SIZE=64798
elif [[ $MODEL_SIZE == "440m" ]];then
    HIDDEN_SIZE=1024
    NUM_HIDDEN_LAYERS=24
    NUM_ATTENTION_HEADS=16
    INTERMEDIATE_SIZE=2816
    ROPE_THETA=10000.0
    MAX_POSITION_EMBEDDINGS=1024
    VOCAB_SIZE=64798
fi

GPT_ARGS=" \
    --hidden_size ${HIDDEN_SIZE} \
    --num_hidden_layers ${NUM_HIDDEN_LAYERS} \
    --num_attention_heads ${NUM_ATTENTION_HEADS} \
    --intermediate_size ${INTERMEDIATE_SIZE} \
    --rope_theta ${ROPE_THETA} \
    --max_position_embeddings ${MAX_POSITION_EMBEDDINGS} \
    --vocab_size ${VOCAB_SIZE} \
"
SCRIPT_ARGS=" \
    --mode ${MODE} \
    --dataset_dir_or_path ${DATASET_DIR_OR_PATH} \
    --resume ${RESUME} \
    --base_model_path ${BASE_MODEL_PATH} \
"

DISTRIBUTED_ARGS=" \
    --nnodes $N_NODES \
    --nproc_per_node $N_GPUS \
"

# 检查num是否大于1
if [ "$N_NODES" -ge 2 ]; then
    DISTRIBUTED_ARGS+=" \
        --node_rank $RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
    "
fi

# 所有参数
ALL_ARGS=" $GPT_ARGS $TRAIN_ARGS $SCRIPT_ARGS "

LAUNCHER="torchrun $DISTRIBUTED_ARGS dpo_train.py "

export CMD="$LAUNCHER $ALL_ARGS"
echo $CMD

killall dpo_train.py

# 执行训练
$CMD 2>&1 | tee ${TRAIN_LOG}

killall dpo_train.py

echo "train end : ${OUTPUT_DIR}"
