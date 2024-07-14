import logging
import numpy as np
import os
import evaluate
import glob
import sys
import math
import json
from dataclasses import dataclass, field
# from itertools import chain
from typing import Optional, List, Dict, Any, Mapping
# from pathlib import Path
import datasets
import torch
import torch.nn as nn
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import LambdaLR
# from datasets import load_dataset, concatenate_datasets, Dataset
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime, timezone
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from transformers.utils.versions import require_version
from sklearn.metrics import accuracy_score
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from configuration_tinyllm import TinyllmConfig
from modeling_tinyllm import TinyllmForCausalLM, TinyllmForSequenceClassification
from tinyllm_dataset import RMDataset
from utils.chatglm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer

@dataclass
class ModelArguments:
    """ 模型相关参数
    """
    hidden_size : Optional[int] = field(
        default=512, 
        metadata={"help": "hidden_size"}
    )
    
    num_hidden_layers : Optional[int] = field(
        default=8, 
        metadata={"help": "num_hidden_layers"}
    )
    
    num_attention_heads : Optional[int] = field(
        default=8, 
        metadata={"help": "transformer num_attention_heads"}
    )
    
    intermediate_size : Optional[int] = field(
        default=1408, 
        metadata={"help": "intermediate_size"}
    )
    
    rope_theta : Optional[float] = field(
        default=10000.0, 
        metadata={"help": "rope_theta"}
    )
    
    max_position_embeddings : Optional[int] = field(
        default=1024, 
        metadata={"help": "max_position_embeddings"}
    )
    
    vocab_size : Optional[int] = field(
        default=64798, 
        metadata={"help": "vocab_size, ref https://github.com/THUDM/ChatGLM3/issues/634"}
    )
    
@dataclass
class ScriptArguments:
    """ 其他相关参数
    """
    mode : Optional[str] = field(
        default="rm", 
        metadata={"help": "save sft *bin file dir"}
    )
    
    dataset_dir_or_path : Optional[str] = field(
        default="data/rm_train", 
        metadata={"help": "save rmtrain *bin file dir"}
    )
    
    resume : Optional[bool] = field(
        default=False, 
        metadata={"help": "use PyTorch 2.0 to compile the model to be faster"}
    )
    
    base_model_path : Optional[str] = field(
        default=" ", 
        metadata={"help": "SFT train, the base model path"}
    )

class RMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """ Define how to compute the reward loss. 
            We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
        """
        rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss

# Define the metric that we'll use for validation.
# accuracy = evaluate.load("accuracy")
# def compute_metrics(eval_pred):
#     predictions, _ = eval_pred
#     # Here, predictions is rewards_j and rewards_k.
#     # We want to see how much of the time rewards_j > rewards_k.
#     # 是这么计算的：
#     # 通过 argmax，得到最大值的 index，当 rewards_j 最大时，返回 0，rewards_k 最大时，返回 1
#     # 正确标签应该是全部为 0（index都在 0 这里）
    
#     # Q: model的输出不是一个score吗，为什么这里可以使用argmax？
#     # A: 下面的 compute_loss 中定义了新的model forward 方法，即会接受两个输入产生两个输出
#     # Trainer 中会把这种两个输出拼起来，从而得到一个在axis=0维度上有两项的形式，因此argmax就是看哪一项更大
#     # 具体可以参考 Trainer 中对 涉及到 compute_loss/logits/training_step/prediction_step 的部分，以及 _gather_and_numpify 方法
#     predictions = np.argmax(predictions, axis=0)
#     labels = np.zeros(predictions.shape)
#     return accuracy.compute(predictions=predictions, references=labels)

from sklearn.metrics import accuracy_score
def compute_metrics(eval_preds):
    predictions = eval_preds.predictions
    preds = np.argmax(predictions, axis=1).reshape(-1)
    labels = np.zeros(preds.shape)
    metric = {
                "accuracy": float(
                    accuracy_score(labels, preds, normalize=True)
                ),
            }
    return metric

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, ScriptArguments, TrainingArguments))
    model_args, script_args, training_args = parser.parse_args_into_dataclasses()
        
    # logger format
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level = logging.WARN,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers = [logging.StreamHandler(sys.stdout)],)
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    
    set_seed(training_args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # init model
    config = transformers.AutoConfig.from_pretrained(
        script_args.base_model_path,
        trust_remote_code=True
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.base_model_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=config.max_position_embeddings
    )

    config.use_cache = False
    config.num_labels = 1
    config.pad_token_id = tokenizer.eos_token_id

    model = TinyllmForSequenceClassification.from_pretrained(
        script_args.base_model_path,
        config=config,
        trust_remote_code=True
    )

    model.to(device)

    ################
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数: {total_params}, {total_params/2**20:.2f}M params")
    logger.info(f"可训练参数: {trainable_params}")
    ##############

    rm_dataset = RMDataset(
        script_args.dataset_dir_or_path, 
        tokenizer, 
        config.max_position_embeddings
    )
    total_len = len(rm_dataset)
    eval_size = int(0.01 * total_len)
    # 划分训练集和验证集
    train_ds, eval_ds = random_split(rm_dataset, [total_len - eval_size, eval_size])

    trainer = RMTrainer(
        model = model,
        args = training_args,
        train_dataset = train_ds,
        eval_dataset = eval_ds,
        compute_metrics=compute_metrics,
    )

    # Training
    trainer.train(script_args.resume)
    torch.save(model.state_dict(),'{}/last_model.pth'.format(training_args.output_dir))
    last_model_dir = os.path.join(training_args.output_dir, 'last_rm_model')
    os.makedirs(last_model_dir, exist_ok=True)
    tokenizer.save_pretrained(last_model_dir)
    # https://github.com/huggingface/transformers/issues/28630
    model.save_pretrained(last_model_dir, safe_serialization=False)

if __name__ == "__main__":
    main()