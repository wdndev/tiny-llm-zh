import logging
import numpy as np
import os
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
from modeling_tinyllm import TinyllmForCausalLM
from tinyllm_dataset import PTMDataset
from utils.chatglm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

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
        default="ptm", 
        metadata={"help": "save pretrain *bin file dir"}
    )
    
    dataset_dir_or_path : Optional[str] = field(
        default="data/pre_train", 
        metadata={"help": "save pretrain *bin file dir"}
    )
    
    resume : Optional[bool] = field(
        default=False, 
        metadata={"help": "use PyTorch 2.0 to compile the model to be faster"}
    )
    
    base_model_path : Optional[str] = field(
        default=" ", 
        metadata={"help": "SFT train, the base model path"}
    )
    
def data_collator_fn(examples):
    # 将所有样本的输入 (`X`) 和标签 (`Y`) 分别堆叠
    input_ids = torch.stack([example[0] for example in examples])
    labels = torch.stack([example[1] for example in examples])

    # 返回一个字典，包含模型需要的键和值
    data_dict = {
        "input_ids": input_ids,
        "labels": labels
    }
    return data_dict
    
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
    gpt_args = dict(
        hidden_size = model_args.hidden_size,
        num_hidden_layers = model_args.num_hidden_layers,
        num_attention_heads = model_args.num_attention_heads,
        intermediate_size = model_args.intermediate_size,
        rope_theta = model_args.rope_theta, 
        max_position_embeddings = model_args.max_position_embeddings,
        vocab_size = model_args.vocab_size, # 64798
    ) 
    gpt_conf = TinyllmConfig(**gpt_args)
    model = TinyllmForCausalLM(gpt_conf)
    model.to(device)
        
    ################
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数: {total_params}, {total_params/2**20:.2f}M params")
    logger.info(f"可训练参数: {trainable_params}")
    ##############
    
    def get_bin_files_abs_paths(directory):
        bin_files_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.bin'):
                    bin_files_paths.append(os.path.abspath(os.path.join(root, file)))
        return bin_files_paths
    # data_path_list = glob.glob(os.path.join(script_args.dataset_dir_or_path, '*.bin'))
    data_path_list = get_bin_files_abs_paths(script_args.dataset_dir_or_path)
    if len(data_path_list) == 0:
        logger.error("***************NO INPUT DATA********************")
    
    train_ds = PTMDataset(data_path_list, max_length = model_args.max_position_embeddings, memmap=False)
    
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_ds,
        # eval_dataset = None,
        # data_collator = data_collator_fn,
    )
    # Training
    trainer.train(script_args.resume)
    torch.save(model.state_dict(),'{}/last_model.pth'.format(training_args.output_dir))
    last_model_dir = os.path.join(training_args.output_dir, 'last_ptm_model')
    os.makedirs(last_model_dir, exist_ok=True)
    # https://github.com/huggingface/transformers/issues/28630
    model.save_pretrained(last_model_dir, safe_serialization=False)
    

if __name__ == "__main__":
    main()

