import logging
import numpy as np
import os
import glob
import sys
import math
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Mapping
import datasets
import torch
import torch.nn as nn
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

from model import Transformer, ModelArgs
from dataset import PretrainDataset, SFTDataset
from utils.chatglm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)



@dataclass
class ModelArguments:
    """ 模型相关参数
    """
    max_seq_len : Optional[int] = field(
        default=512, 
        metadata={"help": "max_seq_len"}
    )
    
    vocab_size : Optional[int] = field(
        default=64793, 
        metadata={"help": "vocab_size"}
    )
    
    dim : Optional[int] = field(
        default=512, 
        metadata={"help": "embedding dim"}
    )
    
    n_layers : Optional[int] = field(
        default=8, 
        metadata={"help": "transformer layers"}
    )
    
    n_heads : Optional[int] = field(
        default=8, 
        metadata={"help": "MHA head"}
    )
    
    n_kv_heads : Optional[int] = field(
        default=8, 
        metadata={"help": "GQA n_kv_heads"}
    )
    
    multiple_of : Optional[int] = field(
        default=32, 
        metadata={"help": "multiple_of"}
    )
    
    dropout : Optional[float] = field(
        default=0.0, 
        metadata={"help": "for pretraining 0 is good, for finetuning try 0.1+"}
    )
    
    use_bias : Optional[bool] = field(
        default=False, 
        metadata={"help": "do we use bias inside LayerNorm and Linear layers?"}
    )
@dataclass
class ScriptArguments:
    """ 其他相关参数
    """
    mode : Optional[str] = field(
        default="pretrain", 
        metadata={"help": "save pretrain *bin file dir"}
    )
    
    dataset_dir : Optional[str] = field(
        default="data/pre_train", 
        metadata={"help": "save pretrain *bin file dir"}
    )
    
    resume : Optional[bool] = field(
        default=False, 
        metadata={"help": "use PyTorch 2.0 to compile the model to be faster"}
    )
    
    is_compile : Optional[bool] = field(
        default=True, 
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
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, script_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
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
        dim = model_args.dim,
        n_layers = model_args.n_layers,
        n_heads = model_args.n_heads,
        n_kv_heads = model_args.n_kv_heads,
        vocab_size = model_args.vocab_size,  #64793
        multiple_of = model_args.multiple_of,
        max_seq_len = model_args.max_seq_len,
        dropout = model_args.dropout,
        use_bias = model_args.use_bias,
    )  # start with model_args from command line
    gpt_conf = ModelArgs(**gpt_args)
    model = Transformer(gpt_conf)
    
    
    model.to(device)
    if script_args.is_compile:
        model = torch.compile(model)
        
    ################
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数: {total_params}, {total_params/2**20:.2f}M params")
    logger.info(f"可训练参数: {trainable_params}")

    # n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    # logger.info(f"Total number of trainable parameters: {n_params:,}")
    # n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    # logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    ##############
    
    def get_bin_files_abs_paths(directory):
        bin_files_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.bin'):
                    bin_files_paths.append(os.path.abspath(os.path.join(root, file)))
        return bin_files_paths
    # data_path_list = glob.glob(os.path.join(script_args.dataset_dir, '*.bin'))
    data_path_list = get_bin_files_abs_paths(script_args.dataset_dir)
    if len(data_path_list) == 0:
        logger.error("***************NO INPUT DATA********************")
    
    train_ds = PretrainDataset(data_path_list, max_length = model_args.max_seq_len, memmap=False)

    
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_ds,
        eval_dataset = None,
        data_collator = data_collator_fn,
    )
    # Training
    trainer.train(script_args.resume)
    torch.save(model.state_dict(),'{}/last_model.pth'.format(training_args.output_dir))
    

if __name__ == "__main__":
    main()

