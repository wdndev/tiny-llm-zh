

from contextlib import nullcontext
import os
import gzip
import shutil
import struct
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


from model import ModelArgs, Transformer
from utils.chatglm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer



def load_checkpoint(args):
    """
        加载给定的模型检查点，并初始化、加载模型权重

        参数：
            - checkpoint (str): 要加载的模型检查点路径

        返回：
            - model (Transformer): 初始化并加载了权重的Transformer模型实例
    """
    # load the provided model checkpoint
    # checkpoint_dict = torch.load(checkpoint, map_location='cpu')
    # gptconf = ModelArgs(**checkpoint_dict['model_args'])
    gpt_args = dict(
        dim = args.dim,
        n_layers = args.n_layers,
        n_heads = args.n_heads,
        n_kv_heads = args.n_kv_heads,
        vocab_size = args.vocab_size,  #64793
        multiple_of = args.multiple_of,
        max_seq_len = args.max_seq_len,
        dropout = args.dropout,
        use_bias = args.use_bias,
    )  # start with model_args from command line
    gpt_conf = ModelArgs(**gpt_args)
    model = Transformer(gpt_conf)
    
    # 移除加载过程中不需要的前缀，并加载模型权重
    # state_dict = checkpoint_dict['model']
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    # 设置模型为评估模式
    model.eval()
    return model

def load_hf_model(model_path):
    """
        加载预训练的Hugging Face模型，并将其转换为自定义Transformer模型

        参数：
            - model_path (str): 预训练Hugging Face模型的路径

        返回：
            - model (Transformer): 转换后的自定义Transformer模型实例
    """
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("Error: transformers package is required to load huggingface models")
        print("Please run `pip install transformers` to install it")
        return None

    # 加载Hugging Face模型
    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    hf_dict = hf_model.state_dict()

    # 将Hugging Face模型配置转换为自定义ModelArgs对象
    config = ModelArgs()
    config.dim = hf_model.config.hidden_size
    config.n_layers = hf_model.config.num_hidden_layers
    config.n_heads = hf_model.config.num_attention_heads
    config.n_kv_heads = hf_model.config.num_attention_heads
    config.vocab_size = hf_model.config.vocab_size
    config.hidden_dim = hf_model.config.intermediate_size
    config.norm_eps = hf_model.config.rms_norm_eps
    config.max_seq_len = hf_model.config.max_position_embeddings

    # 创建一个基于上述配置的自定义Transformer模型
    model = Transformer(config)

    model.tok_embeddings.weight = nn.Parameter(hf_dict['model.embed_tokens.weight'])
    model.norm.weight = nn.Parameter(hf_dict['model.norm.weight'])

    # 将Hugging Face模型的权重赋值给自定义Transformer模型的Embedding层和Norm层
    def permute_reverse(w, n_heads=config.n_heads, dim1=config.dim, dim2=config.dim):
        return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    for layer in model.layers:
        i = layer.layer_id
        layer.attention_norm.weight = nn.Parameter(hf_dict[f'model.layers.{i}.input_layernorm.weight'])
        layer.attention.wq.weight = nn.Parameter(permute_reverse(hf_dict[f'model.layers.{i}.self_attn.q_proj.weight']))
        layer.attention.wk.weight = nn.Parameter(permute_reverse(hf_dict[f'model.layers.{i}.self_attn.k_proj.weight']))
        layer.attention.wv.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.v_proj.weight'])
        layer.attention.wo.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.o_proj.weight'])
        layer.ffn_norm.weight = nn.Parameter(hf_dict[f'model.layers.{i}.post_attention_layernorm.weight'])
        layer.feed_forward.w1.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.gate_proj.weight'])
        layer.feed_forward.w2.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.down_proj.weight'])
        layer.feed_forward.w3.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.up_proj.weight'])

    # 设置最终分类器的权重
    model.output.weight = nn.Parameter(hf_dict['lm_head.weight'])
    
    # 设置模型为评估模式
    model.eval()
    return model

def generate(data:list, model, tokenizer, device, args):
    ctx = nullcontext() if device == 'cpu' else torch.cuda.amp.autocast()
    ans_list = []
    for p in data[:100]:
        # run generation
        prompt=p['question']
        if args.eval_type == 'ptm':
            x = tokenizer.encode(prompt, add_special_tokens=False)
        else:
            x = tokenizer.encode(prompt, add_special_tokens=False) + [tokenizer.special_tokens['<bos>']]
        
        # print(tokenizer.decode(x))
        
        x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])
        with torch.no_grad():
            with ctx:
                y = model.generate( x,
                                    tokenizer.special_tokens['<eos>'], 
                                    max_new_tokens=args.max_new_tokens, 
                                    temperature=args.temperature, 
                                    top_k=args.top_k)
                answer = tokenizer.decode(y[0].tolist())
                answer = answer.replace(prompt,'')
                ans_list.append(answer)
                print('[prompt]:',prompt)
                print('[answer]:',answer)
                print('---------------')
    return ans_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_type", default="ptm", type=str, help="eval type")
    # model args
    parser.add_argument("--dim", default=512, type=int, help="model dim")
    parser.add_argument("--n_layers", default=8, type=int, help="model n_layers")
    parser.add_argument("--n_heads", default=8, type=int, help="model n_heads")
    parser.add_argument("--n_kv_heads", default=8, type=int, help="model n_kv_heads")
    parser.add_argument("--multiple_of", default=32, type=int, help="model multiple_of")
    parser.add_argument("--dropout", default=0.1, type=float, help="model dropout")
    parser.add_argument("--use_bias", default=False, type=bool, help="model use_bias")
    parser.add_argument("--vocab_size", default=64793, type=int, help="model vocab_size")
    parser.add_argument("--max_seq_len", default=512, type=int, help="model max_seq_len")
    # generate arge
    parser.add_argument("--max_new_tokens", default=100, type=int, help="generate max_new_tokens")
    parser.add_argument("--temperature", default=1.0, type=float, help="generate temperature")
    parser.add_argument("--top_k", default=30, type=int, help="generate top_k")
    # load model type
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="model checkpoint, .pt file")
    group.add_argument("--hf", type=str, help="huggingface model path")
    # tokenizer
    parser.add_argument("--tokenizer_path", default="utils/chatglm3_tokenizer/tokenizer.model", type=str, help="tokenizer path")
    args = parser.parse_args()
    if args.checkpoint:
        model = load_checkpoint(args)
    elif args.hf:
        model = load_hf_model(args.hf)

    if model is None:
        parser.error("Can't load input model!")

    # seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    
    model.to(device)
    
    tokenizer=ChatGLMTokenizer(vocab_file=args.tokenizer_path)
    
    
    # data = [
    #     {"question": "床前明月光，疑是地上霜。举头望明月，"},
    #     {"question": "请你讲一个童话故事："},
    #     {"question": "《小王子》是一本畅销童话书，它讲述了："},
    # ]

    data = [
        {"question": "三十年河东，三十年河西，莫欺少年穷，"},
        {"question": "人生如棋，落子无悔，"},
        {"question": "彼岸花开开彼岸，断肠草愁愁断肠，奈何桥前可奈何，"},
        {"question": "散在青春里一场梦，"},
    ]
    
    
    # generate
    generate(data, model, tokenizer, device, args)


