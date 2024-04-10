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


# 常见参数
N_NODES=1   # 节点数
N_GPUS=8    # 每个节点GPU数目
MBS=32       # 单卡bs
GAS=1       # 梯度累积
GRAD_CLIP=1     # 梯度裁剪

# 学习率初始化
LR=3e-4 
LR_SCHEDULER_TYPE="cosine"
WARMUP_RATION=0.05

TRAIN_EPOCHS=5          # 训练轮次
LOGGING_STEPS=100       # 记录日志步数
CKPT_SAVE_STEPS=10000    # ckpt保存步数

SEED=12
DS_DTYPE="fp16" # [fp16, bf16]
IS_COMPILE="True"
RESUME="False"

# 数据
MODE="sft" # [ptm, sft, rm, rlhf]
DATASET_DIR="data/sft_train"
BASE_MODEL_PATH="/wangdongnian/personal/tiny-llama2.zh/outputs/ckpt/ptm_tiny_llama2_9m_epoch3/checkpoint-1010000/pytorch_model.bin"

MODEL_SIZE="9m" # [9m, 24m, 58m, 134m]
MODEL_NAME="${MODE}_tiny_llama2_${MODEL_SIZE}"
OUTPUT_DIR="outputs/ckpt/${MODEL_NAME}_epoch${TRAIN_EPOCHS}"
mkdir -p $OUTPUT_DIR
TRAIN_LOG="${OUTPUT_DIR}/train_$(date "+%Y%m%d%H%M").log"
# tensorboard输出路径
TB_DIR="outputs/tensorboard/${MODEL_NAME}"
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
    --save_total_limit 20 \
    --save_steps ${CKPT_SAVE_STEPS} \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --save_safetensors False \
    --ddp_find_unused_parameters False \
    "

if [[ $MODEL_SIZE == "9m" ]];then
    MAX_SEQ_LEN=512
    DIM=120
    N_LAYERS=6
    N_HEADS=6
    N_KV_HEADS=6
    MULTIPLE_OF=32
    DROPOUT=0.0
    VOCAB_SIZE=64793
    BATCH_SIZE=64
elif [[ $MODEL_SIZE == "24m" ]];then
    MAX_SEQ_LEN=512
    DIM=288
    N_LAYERS=6
    N_HEADS=6
    N_KV_HEADS=6
    MULTIPLE_OF=32
    DROPOUT=0.1
    VOCAB_SIZE=64793
    BATCH_SIZE=48
elif [[ $MODEL_SIZE == "58m" ]];then
    MAX_SEQ_LEN=512
    DIM=512
    N_LAYERS=8
    N_HEADS=8
    N_KV_HEADS=8
    MULTIPLE_OF=32
    DROPOUT=0.1
    VOCAB_SIZE=64793
    BATCH_SIZE=32
elif [[ $MODEL_SIZE == "134m" ]];then
    MAX_SEQ_LEN=1024
    DIM=768
    N_LAYERS=12
    N_HEADS=12
    N_KV_HEADS=12
    MULTIPLE_OF=32
    DROPOUT=0.1
    VOCAB_SIZE=64793
    BATCH_SIZE=32
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
SCRIPT_ARGS=" \
    --mode ${MODE} \
    --dataset_dir ${DATASET_DIR} \
    --is_compile ${IS_COMPILE} \
    --resume ${RESUME} \
    --base_model_path ${BASE_MODEL_PATH} \
    "

# 所有参数
ALL_ARGS=" $GPT_ARGS $TRAIN_ARGS $SCRIPT_ARGS "

LAUNCHER="torchrun --nnodes $N_NODES --nproc_per_node $N_GPUS sft_train.py "

export CMD="$LAUNCHER $ALL_ARGS"
echo $CMD

killall sft_train.py

# 执行训练
$CMD 2>&1 | tee ${TRAIN_LOG}

killall sft_train.py

echo "train end : ${OUTPUT_DIR}"
# nohup torchrun --standalone --nproc_per_node=$N_GPUS pretrain.py \
#                 --out_dir="$OUTPUT_DIR/$MODEL_NAME"   \
#                 --vocab_size=$VOCAB_SIZE    \
#                 --max_seq_len=$VOCAB_SIZE   \
#                 --dim=$DIM                  \
#                 --n_layers=$N_LAYERS        \
#                 --n_heads=$N_HEADS          \
#                 --n_kv_heads=$N_KV_HEADS    \
#                 --multiple_of=$MULTIPLE_OF  \
#                 --dropout=$DROPOUT          \
#                 --batch_size=$BATCH_SIZE    \
#                 >> $log_file 2>&1 &

