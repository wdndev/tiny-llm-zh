"""
该脚本包含用于模型导出的功能和工具。基本上，我们有一系列模型版本，并且希望将它们导出为.bin文件，以便在C语言环境中读取和推理。

在PyTorch文件/模型的“输入”版本中：
    - Meta官方发布的Llama 2权重
    - Huggingface库中提供的权重
    - 本仓库训练的llama2.c模型

在.bin文件的“输出”版本中：
    - v0：原始llama2.c仓库的遗留文件（最终将被弃用）
    - v1-vN：经过改进的.bin文件，具有适当的头部信息、缓存对齐等特性

该脚本旨在提供所有这些转换功能。
"""
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

# -----------------------------------------------------------------------------
# common utilities

def serialize_fp32(file, tensor):
    """ 将一个fp32格式的张量写入已以二进制写模式打开的文件中
    """
    # 将张量从计算图分离并转换到CPU上，降维为一维，并转为float32类型numpy数组
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    # 使用struct模块打包成浮点数格式的二进制数据
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def serialize_int8(file, tensor):
    """ 将一个int8格式的张量写入已以二进制写模式打开的文件中
    """
    # 将张量从计算图分离并转换到CPU上，降维为一维，并转为int8类型的numpy数组
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    # 使用struct模块打包成整数格式的二进制数据
    b = struct.pack(f'{len(d)}b', *d)
    file.write(b)

def quantize_q80(w, group_size):
    """
    输入一个张量，返回其Q8_0量化版本
    即采用对称量化策略将其转化为int8类型，范围在[-127, 127]
    """
    # 确保张量元素数量能被group_size整除
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    # 将张量转换为float32类型
    w = w.float()
    # 将张量重塑为每一组(group_size)为一行的形式
    w = w.reshape(-1, group_size)
    # 在每一组中找到绝对值的最大值
    wmax = torch.abs(w).max(dim=1).values
    # 计算缩放因子，使得float数值 = 量化后的整数 * 缩放因子
    scale = wmax / 127.0
    # 将张量按比例缩放到[-127, 127]范围内
    quant = w / scale[:,None]
    # 四舍五入到最近的整数，转换为int8张量
    int8val = torch.round(quant).to(torch.int8)
    
    # 对量化后的整数张量进行反量化，通过乘回缩放因子得到近似的浮点数
    fp32val = (int8val.float() * scale[:,None]).view(-1)
    # 将反量化后的浮点数恢复为原始形状
    fp32valr = fp32val.reshape(-1, group_size)
    # 计算每一组内的最大误差（量化前后的差值）
    err = torch.abs(fp32valr - w).max(dim=1).values
    # 找出所有组中的最大误差
    maxerr = err.max().item()
    
    # 返回量化后的int8张量、缩放因子和最大误差
    return int8val, scale, maxerr

# -----------------------------------------------------------------------------
# legacy

def legacy_export(model, filepath):
    """ 用于原始方式导出llama2.c的bin文件，即v0版本格式
    """
    # 打开输出文件，以二进制写模式
    out_file = open(filepath, 'wb')

    # 首先写入头部信息
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    p = model.args
    # 判断是否共享分类器权重
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    
    # 对于遗留格式，使用负的/正的词汇表大小作为共享分类器标志
    if not shared_classifier:
        p.vocab_size = -p.vocab_size
    # 获取n_kv_heads参数
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    # 封装头部结构体数据
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                    n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)

    # 接下来写入嵌入权重: 将词嵌入权重序列化并写入文件
    serialize_fp32(out_file, model.tok_embeddings.weight)

    # 现在写入所有层的权重
    # 注意力权重
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention_norm.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wq.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wk.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wv.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wo.weight)
    # 前馈神经网络（FFN）权重
    for layer in model.layers:
        serialize_fp32(out_file, layer.ffn_norm.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w1.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w2.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w3.weight)
    # 最终的根均方归一化（rmsnorm）权重
    serialize_fp32(out_file, model.norm.weight)
    # freqs_cis
    serialize_fp32(out_file, model.freqs_cos[:p.max_seq_len])
    serialize_fp32(out_file, model.freqs_sin[:p.max_seq_len])

    # 最终分类器权重（仅当未共享分类器权重时写入）
    if not shared_classifier:
        serialize_fp32(out_file, model.output.weight)

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")

# -----------------------------------------------------------------------------
# new version

def version1_export(model, filepath):
    """
    将模型权重以全精度浮点数形式导出到C语言可读的.bin文件。
    此功能与legacy_export相同，但带有适当的头部信息。
    """
    version = 1

    out_file = open(filepath, 'wb')
    
    # 首先写出头部信息，头部总共占用256字节
    # 1) 写入魔数，为ASCII表示的"ak42"的uint32值
    out_file.write(struct.pack('I', 0x616b3432))
    # 2) 写入版本号，为整型值
    out_file.write(struct.pack('i', version))
    # 3) 写入参数，共7个整型值
    p = model.args
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                    n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)
    # 4) 写入其他标志位
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    # 是否共享分类器权重标志位
    out_file.write(struct.pack('B', int(shared_classifier)))
    pad = 256 - out_file.tell() # 补齐剩余部分至256字节，tell()返回当前位置
    assert pad >= 0
    # 用零填充剩余部分
    out_file.write(b'\0' * pad)

    # 现在写出所有参数
    weights = [
        *[layer.attention_norm.weight for layer in model.layers],
        *[layer.ffn_norm.weight for layer in model.layers],
        model.norm.weight,
        model.tok_embeddings.weight,
        *[layer.attention.wq.weight for layer in model.layers],
        *[layer.attention.wk.weight for layer in model.layers],
        *[layer.attention.wv.weight for layer in model.layers],
        *[layer.attention.wo.weight for layer in model.layers],
        *[layer.feed_forward.w1.weight for layer in model.layers],
        *[layer.feed_forward.w2.weight for layer in model.layers],
        *[layer.feed_forward.w3.weight for layer in model.layers],
    ]
    # 若不共享分类器权重，则添加输出权重
    if not shared_classifier:
        weights.append(model.output.weight)
    for w in weights:
        serialize_fp32(out_file, w)

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")

def version2_export(model, filepath, group_size=64):
    """
    将模型权重以Q8_0量化格式导出到C语言可读的.bin文件。
    具体步骤如下：
    - 将所有权重量化为有符号整8位，范围在[-127, 127]
    - 其他张量（rmsnorm参数）仍保留并以全精度浮点数格式导出
    - 量化过程中按group_size分组以减小异常值的影响
    """
    version = 2

    # 验证导出类型的相关条件
    while model.args.dim % group_size != 0:
        group_size //= 2
        print(f"BACKOFF: reducing group size to {group_size} to fit hidden_dim")
    weights = [
        model.tok_embeddings.weight,
        *[layer.attention.wq.weight for layer in model.layers],
        *[layer.attention.wk.weight for layer in model.layers],
        *[layer.attention.wv.weight for layer in model.layers],
        *[layer.attention.wo.weight for layer in model.layers],
        *[layer.feed_forward.w1.weight for layer in model.layers],
        *[layer.feed_forward.w2.weight for layer in model.layers],
        *[layer.feed_forward.w3.weight for layer in model.layers],
    ]
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    if not shared_classifier:
        weights.append(model.output.weight)
     # 验证权重张量的元素数量能否被group_size整除
    for w in weights:
        assert w.numel() % group_size == 0, f"weight {i} has numel {w.numel()}, not a multiple of group_size {group_size}"

    # 开始写入文件
    out_file = open(filepath, 'wb')
    
    # 写出头部信息，同样为256字节
    # 1) 写入魔数，为ASCII表示的"ak42"的uint32值
    out_file.write(struct.pack('I', 0x616b3432))
    # 2) 写入版本号，为整型值
    out_file.write(struct.pack('i', version))
    # 3) 写入参数，共7个整型值
    p = model.args
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                    n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)
    # 4) 写入其他标志位
    out_file.write(struct.pack('B', int(shared_classifier)))    # 是否共享分类器权重标志位
    out_file.write(struct.pack('i', group_size)) # 用于量化的分组大小
    pad = 256 - out_file.tell() # 补齐剩余部分至256字节
    assert pad >= 0
    out_file.write(b'\0' * pad)
    # 头部信息写完后，开始写出模型参数

    # 首先写出所有保留全精度浮点数格式的参数，即各个归一化参数
    for layer in model.layers: # attention norms
        serialize_fp32(out_file, layer.attention_norm.weight)
    for layer in model.layers: # MLP norms
        serialize_fp32(out_file, layer.ffn_norm.weight)
    serialize_fp32(out_file, model.norm.weight) # final pre-classifier norm

    # 现在写出所有要量化为Q8_0格式的参数
    # note we skip classifier weights, which are shared with the embedding
    ew = []     # 用于记录量化误差
    for i, w in enumerate(weights):
        # 对当前权重进行量化
        q, s, err = quantize_q80(w, group_size)
        # 将量化后的int8权重保存到文件
        serialize_int8(out_file, q) # 保存整型权重张量
        serialize_fp32(out_file, s) # 保存缩放因子
        # logging
        ew.append((err, w.shape))
        print(f"{i+1}/{len(weights)} quantized {tuple(w.shape)} to Q8_0 with max error {err}")

    # print the highest error across all weights, should be very small, e.g. O(~0.001)
    # 打印所有权重的最大量化误差，应非常小，例如O(~0.001)
    ew.sort(reverse=True)
    print(f"max quantization group error across all weights: {ew[0][0]}")

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")

def hf_export(llama_model, filepath, group_size=64, dtype=torch.float32):
    """ 
    生成HuggingFace模型对应的pytorch_model.bin状态字典和config.json文件

    参数：
    - llama_model：LLAMA模型实例
    - filepath：保存文件的目标目录
    - group_size（默认值=64）：量化时使用的分组大小
    - dtype（默认值=torch.float32）：模型权重的数据类型

    函数首先尝试导入HuggingFace LlamaConfig，然后根据LLAMA模型实例，转换并重组权重，
    最后生成并保存状态字典和配置文件至指定目录。
    """

    try:
        from transformers.models.llama.configuration_llama import LlamaConfig
    except ImportError:
        print("Error: transformers package is required to load huggingface models")
        print("Please run `pip install transformers` to install it")
        return None

    # 生成HuggingFace模型的 state_dict
    hf_state_dict = {}

    # 根据LLAMA模型参数计算相关变量
    dim = llama_model.params.dim
    num_key_value_heads = llama_model.params.n_kv_heads
    n_rep = llama_model.params.n_heads // num_key_value_heads
    key_value_dim = dim // n_rep

    # 定义用于转换权重顺序的辅助函数
    # See: https://github.com/huggingface/transformers/blob/b132c1703eb1c8bd9dfa4ad6a9be2bfd6ef819e9/src/transformers/models/llama/convert_llama_weights_to_hf.py#L122
    def permute_original(w, n_heads=llama_model.params.n_heads, dim1=dim, dim2=dim):
        return w.view(dim1, dim2).reshape(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    # Add each layer's weights to the HF state dictionary
    hf_state_dict['model.embed_tokens.weight'] = llama_model.tok_embeddings.weight.clone().to(dtype)
    hf_state_dict['model.norm.weight'] = llama_model.norm.weight.clone().to(dtype)

    # 遍历每一层并将各层权重添加至HF状态字典
    for i, layer in enumerate(llama_model.layers):
        layer_id = layer.layer_id
        hf_state_dict[f'model.layers.{i}.input_layernorm.weight'] = llama_model.layers[layer_id].attention_norm.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.self_attn.q_proj.weight'] = permute_original(llama_model.layers[layer_id].attention.wq.weight.clone()).to(dtype)
        hf_state_dict[f'model.layers.{i}.self_attn.k_proj.weight'] = permute_original(llama_model.layers[layer_id].attention.wk.weight.clone(), num_key_value_heads, key_value_dim, dim).to(dtype)
        hf_state_dict[f'model.layers.{i}.self_attn.v_proj.weight'] = llama_model.layers[layer_id].attention.wv.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.self_attn.o_proj.weight'] = llama_model.layers[layer_id].attention.wo.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.post_attention_layernorm.weight'] = llama_model.layers[layer_id].ffn_norm.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.mlp.gate_proj.weight'] = llama_model.layers[layer_id].feed_forward.w1.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.mlp.down_proj.weight'] = llama_model.layers[layer_id].feed_forward.w2.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.mlp.up_proj.weight'] = llama_model.layers[layer_id].feed_forward.w3.weight.clone().to(dtype)

    # llama2.c usually uses tied weights -> reference the embed_tokens.weights instead
    # 如果使用了绑定了词嵌入权重，则引用embed_tokens.weights
    hf_state_dict['lm_head.weight'] = hf_state_dict['model.embed_tokens.weight']

    # 检查词嵌入权重是否绑定，如果不绑定则使用手动输出权重
    _embeddings_are_tied: bool = torch.equal(llama_model.tok_embeddings.weight, llama_model.output.weight)
    if not _embeddings_are_tied:
        hf_state_dict['lm_head.weight'] = llama_model.output.weight.clone().to(dtype)


    # 生成LlamaConfig（对应transformers.models.llama.configuration_llama）

    # 从llama.c模型中提取必要的属性
    vocab_size = llama_model.params.vocab_size
    hidden_size = llama_model.params.dim
    intermediate_size = llama_model.layers[0].feed_forward.w1.weight.shape[0]
    num_hidden_layers = llama_model.params.n_layers
    num_attention_heads = llama_model.params.n_heads
    num_key_value_heads = llama_model.params.n_kv_heads
    max_position_embeddings = llama_model.params.max_seq_len
    rms_norm_eps = llama_model.params.norm_eps

    # TODO: 需检查以下参数的值：pretraining_tp, initializer_range, use_cache,
    #       rope_theta, 和 rope_scaling.

    # 创建LlamaConfig实例
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        tie_word_embeddings=_embeddings_are_tied,
        # Manual
        architectures=["LlamaForCausalLM"],
        hidden_act="silu",
    )


    # 将文件保存至目标目录
    # 若目录不存在则先创建
    os.makedirs(filepath, exist_ok=True)

    # 将状态字典以.bin格式保存，并将配置文件保存为.json格式
    torch.save(hf_state_dict, os.path.join(filepath, "pytorch_model.bin"))
    config.save_pretrained(filepath)


# -----------------------------------------------------------------------------
# Load / import functions

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

def load_meta_model(model_path):
    """
        加载meta模型，该模型由多个子模型合并而成

        参数：
            - model_path (str): meta模型所在目录路径

        返回：
            - model (Transformer): 合并后的新Transformer模型实例
    """
    params_path = os.path.join(model_path, 'params.json')
    with open(params_path) as f:
        params = json.load(f)
        print(params)

    # 加载所有子模型检查点
    model_paths = sorted(list(Path(model_path).glob('consolidated.*.pth')))
    models = [torch.load(p, map_location='cpu') for p in model_paths]

    # 定义函数，拼接子模型的权重
    def concat_weights(models):
        state_dict = {}
        for name in list(models[0]):
            # 收集每个名字对应的张量
            tensors = [model[name] for model in models]
            # 单一张量或者1维张量直接添加到state_dict
            if len(tensors) == 1 or len(tensors[0].shape) == 1:
                state_dict[name] = tensors[0]
                continue
            # 根据张量名称确定拼接轴
            is_axis_1 = (
                name.startswith('tok_embeddings.')
                or name.endswith('.attention.wo.weight')
                or name.endswith('.feed_forward.w2.weight')
            )
            axis = 1 if is_axis_1 else 0
            # 沿指定轴拼接张量
            state_dict[name] = torch.cat(tensors, dim=axis)
            # 删除已拼接的张量，释放内存
            for model in models:
                del model[name]
        return state_dict

    # 拼接子模型权重
    state_dict = concat_weights(models)
    del models

    # 设置ModelArgs配置参数
    config = ModelArgs()
    config.dim = params["dim"]
    config.n_layers = params["n_layers"]
    config.n_heads = params["n_heads"]
    config.n_kv_heads = params.get('n_kv_heads') or params['n_heads']
    config.multiple_of = params["multiple_of"]
    config.norm_eps = params["norm_eps"]

    config.vocab_size = state_dict['tok_embeddings.weight'].shape[0]
    config.max_seq_len = 2048


    # 创建新的Transformer模型实例并逐个设置权重
    model = Transformer(config)

    # 设置Embedding层与Norm层权重
    model.tok_embeddings.weight = nn.Parameter(state_dict['tok_embeddings.weight'])
    model.norm.weight = nn.Parameter(state_dict['norm.weight'])

    # 遍历每一层，逐个设置注意力机制及全连接层的权重
    for layer in model.layers:
        i = layer.layer_id
        layer.attention_norm.weight = nn.Parameter(state_dict[f'layers.{i}.attention_norm.weight'])
        layer.attention.wq.weight = nn.Parameter(state_dict[f'layers.{i}.attention.wq.weight'])
        layer.attention.wk.weight = nn.Parameter(state_dict[f'layers.{i}.attention.wk.weight'])
        layer.attention.wv.weight = nn.Parameter(state_dict[f'layers.{i}.attention.wv.weight'])
        layer.attention.wo.weight = nn.Parameter(state_dict[f'layers.{i}.attention.wo.weight'])
        layer.ffn_norm.weight = nn.Parameter(state_dict[f'layers.{i}.ffn_norm.weight'])
        layer.feed_forward.w1.weight = nn.Parameter(state_dict[f'layers.{i}.feed_forward.w1.weight'])
        layer.feed_forward.w2.weight = nn.Parameter(state_dict[f'layers.{i}.feed_forward.w2.weight'])
        layer.feed_forward.w3.weight = nn.Parameter(state_dict[f'layers.{i}.feed_forward.w3.weight'])

    # 设置最终分类器权重
    model.output.weight = nn.Parameter(state_dict['output.weight'])
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


# -----------------------------------------------------------------------------
# API entrypoint

def model_export(model, filepath, version, dtype=torch.float32):
    """
    版本说明：
        - v-1: 适用于Hugging Face的导出方式，即将模型转换为可在HF外部使用的格式
        - v0: 已弃用的legacy llama2.c 浮点数格式
        - v1: 浮点32位导出
        - v2: 类似于llama.cpp的int8量化Q8_0格式导出，采用分组量化
        - TODO: 添加对其他版本不同数据类型的导出支持
    """

    if version == 0:
        legacy_export(model, filepath)
    elif version == 1:
        version1_export(model, filepath)
    elif version == 2:
        version2_export(model, filepath)
    elif version == -1:
        hf_export(model, filepath, dtype)
    else:
        raise ValueError(f"unknown version {version}")

def torchscript_export(model, filepath, zero_params=False, gzip_output=False):
    """
    (This was submitted via a PR earlier. Leaving it here, but "orphaned" for now)
    Saves the model as a TorchScript.
    The resulting file can be loaded in C++ code and then used for training or
    inference with:
        #include <torch/script.h>
        torch::jit::Module module = torch::jit::load("model.pt")
    Note that the serialized model includes the initial parameters and with the default
    ModelArgs the file is 59M and gzips down to 55M. If you want to serialize/distribute
    the model parameters separately you can zero out the parameters before saving it and
    it will gzip down to 780K.
    """

    # If requested zero params before saving the model. This is useful in
    # conjunction with gzip_output.
    if zero_params:
        for p in model.parameters():
            p.detach().zero_()

    torch.jit.save(torch.jit.script(model), filepath)

    if gzip_output:
        with open(filepath, "rb") as f_in:
            with gzip.open(f"{filepath}.gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.unlink(filepath)

# -----------------------------------------------------------------------------
# CLI entrypoint

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
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
    # output args
    parser.add_argument("--output_dir", type=str, help="the output filepath")
    # control args
    parser.add_argument("--version", default=0, type=int, help="the version to export with")
    parser.add_argument("--dtype", type=str, help="dtype of the model (fp16, fp32)", default="fp32")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="model checkpoint, .pt file")
    group.add_argument("--meta-llama", type=str, help="meta llama model path")
    group.add_argument("--hf", type=str, help="huggingface model path")
    args = parser.parse_args()
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    if args.checkpoint:
        model = load_checkpoint(args)
    elif args.meta_llama:
        model = load_meta_model(args.meta_llama)
    elif args.hf:
        model = load_hf_model(args.hf)

    if model is None:
        parser.error("Can't load input model!")

    # export
    model_export(model, args.output_dir, args.version, args.dtype)
