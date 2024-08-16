"""
量化问题：
https://huggingface.co/astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit/discussions/5
https://github.com/AutoGPTQ/AutoGPTQ/issues/657
"""
import os
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer, HfArgumentParser

# Assuming 'read_jsonl_file' is a function defined in 'utilis' module
from utilis import read_jsonl_file

import logging

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    # Basic settings
    model_id: Optional[str] = field(default="", metadata={"help": "The location of the SFT model name or path."})
    # {"input": question, "target": answer}
    dataset_dir_or_path: Optional[str] = field(default="", metadata={"help": "The location of the dataset directory or path."})
    quant_output_dir: Optional[str] = field(default="./results", metadata={"help": "The output directory for the quantized model."})
    ngpus: Optional[int] = field(default=1, metadata={"help": "Number of GPUs for quantization."})
    gpu_max_memory: Optional[int] = field(default=20, metadata={"help": "Max memory per GPU for quantization (in GB)."})
    
    # GPTQ parameters
    bits: Optional[int] = field(default=4, metadata={"help": "Quantization bits (4 or 8)."})
    group_size: Optional[int] = field(default=128, metadata={"help": "Group size for quantization (32, 64, 128)."})
    damp_percent: Optional[float] = field(default=0.1, metadata={"help": "Damping percentage for quantization (0.1, 0.01)."})
    desc_act: Optional[bool] = field(default=False, metadata={"help": "Whether to use descending activation (False speeds up inference but may affect perplexity)."})
    static_groups: Optional[bool] = field(default=False, metadata={"help": "Whether to use static groups for quantization."})
    sym: Optional[bool] = field(default=True, metadata={"help": "Whether to use symmetric quantization."})
    true_sequential: Optional[bool] = field(default=True, metadata={"help": "Whether to use true sequential quantization."})
    
    # Training parameters
    max_len: Optional[int] = field(default=8192, metadata={"help": "Maximum length of input data."})
    batch_size: Optional[int] = field(default=1, metadata={"help": "Batch size for quantization training."})
    cache_examples_on_gpu: Optional[bool] = field(default=False, metadata={"help": "Whether to cache examples on GPU during quantization."})
    use_triton: Optional[bool] = field(default=False, metadata={"help": "Whether to use Triton for quantization."})

def data_process(data_list, max_len, tokenizer: AutoTokenizer):
    def qwen_process(item):
        input_text = item["input"]
        target_text = item["target"]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": target_text}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        model_inputs = tokenizer(text, truncation=True, padding='max_length', max_length=max_len)
        input_ids = torch.tensor(model_inputs['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(model_inputs['attention_mask'], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    return [qwen_process(item) for item in data_list]

def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    quantize_config = BaseQuantizeConfig(
        bits=script_args.bits,  # 4 or 8
        group_size=script_args.group_size,
        damp_percent=script_args.damp_percent,
        desc_act=script_args.desc_act,  # set to False can significantly speed up inference but the perplexity may slightly bad
        static_groups=script_args.static_groups,
        sym=script_args.sym,
        true_sequential=script_args.true_sequential
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_id,
        trust_remote_code=True
    )
    
    model = AutoGPTQForCausalLM.from_pretrained(
        script_args.model_id,
        quantize_config,
        max_memory={i: f"{script_args.gpu_max_memory}GB" for i in range(script_args.ngpus)}
    )
    
    data_list = read_jsonl_file(script_args.dataset_dir_or_path)
    quant_data = data_process(
        data_list, 
        script_args.max_len,
        tokenizer
    )
    
    model.quantize(
        quant_data,
        cache_examples_on_gpu=script_args.cache_examples_on_gpu,
        batch_size=script_args.batch_size,
        use_triton=script_args.use_triton
    )
    
    model.save_quantized(script_args.quant_output_dir, use_safetensors=True)
    tokenizer.save_pretrained(script_args.quant_output_dir)

if __name__ == "__main__":
    main()