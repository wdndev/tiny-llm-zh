
import math
import inspect
from typing import Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

@dataclass
class ModelArgs:
    # 默认参数是 llama 7b
    dim : int = 4096
    n_layers : int = 32
    n_heads : int = 32
    n_kv_heads : Optional[int] = None
    vocab_size : int = 32000
    hidden_dim : Optional[int] = None
    multiple_of : int = 256  # MLP隐藏层的倍数
    norm_eps : float = 1e-5
    max_seq_len = 2048
    dropout : float = 0.0       

class RMSNorm(nn.Module):
    """ root mean square(RMS)归一化
    """
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """ 预计算CIS(正弦和余弦)频率矩阵
        - dim : CIS频率的维数
        - end : CIS频率的最大索引
        - theta : CIS频率的比例因子。默认为10000.0。
    """
    # 计算频率向量，根据逆频率公式，频率呈线性衰减分布
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """ reshape 频率张量形状，用于广播张量x
        - freqs_cis
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(xq: torch.Tensor,
                     xk: torch.Tensor,
                     freqs_cos: torch.Tensor, 
                     freqs_sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ 旋转位置编码
        - xq : (b, l, qhn, hd) -> (batch_size, length, q_head_num, hidden_dim)
        - xk : (b, l, kvhn, hd)
        - freqs_cos : (l, hd//2)
        - freqs_sin : (l, hd//2)
    """
    # reshape xq,xk 用复数表示
    # (b, l, qhn, hd//2) 
    # (b, l, kvhn, hd//2)
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)
    
    # reshape freqs_cos and freqs_sin
    # (1, l, 1, hd//2)
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)
    
    # apply rotation using real numbers
    # (b, l, qhn, hd//2)
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    # (b, l, kvhn, hd//2)
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    # (b, l, qhn, hd)
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    # (b, l, kvhn, hd)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """ torch.repeat_interleave(x, dim=2, repeats=n_rep)
    """
    bs, sl, n_kn_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # (b, l, qhn, hd)
    z = x[:, :, :, None, :].expand(bs, sl, n_kn_heads, n_rep, head_dim).reshape(bs, sl, n_kn_heads * n_rep, head_dim)
    return z

class Attention(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        # 参数
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        
        # use flash attention or a manual implementation?
        # 判断是否使用内置的Flash Attention（PyTorch版本要求>=2.0）
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)
    
    def forward(self,
                x: torch.Tensor, 
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor):
        """ 前向传播
            - x : (b, l, d)  -> (batch_size, seq_len, dim)
            - freqs_cos : (l, hd//2)
            - freqs_sin : (l, hd//2)
        """
        bsz, seqlen, _ = x.shape
        
        # qkv
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        
        # 旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        
        # GQA, 扩展k，v输出
        # (bs, seqlen, n_local_heads, head_dim)
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xk, self.n_rep)
        
        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # flash implementation
        if self.flash:
            # 使用内置的scaled_dot_product_attention函数实现注意力机制
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # manual implementation
            # 如果没有内置实现，手动计算注意力得分并应用掩码
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]   # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
    
        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

class FeedForward(nn.Module):
    """ FFN
    """
    def __init__(self, dim: int, hidden_dim:int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
class TransformerBlock(nn.Module):
    """ Transformer Block
    """
    def __init__(self, layer_id: int, args: ModelArgs) -> None:
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
    def forward(self, x, freqs_cos, freqs_sin):
        # (b, l, d)
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
    
    
class TinyLlama2(nn.Module):
    last_loss: Optional[torch.Tensor]
    
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(layer_id, args))
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        # embedding 与 unembedding 参数共享
        # https://paperswithcode.com/method/weight-tying
        self.tok_embeddings.weight = self.output.weight
        
        # RoPE相对位置嵌入的预计算
        freqs_cos, freqs_sin = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        
        # 初始化模型
        self.apply(self._init_weights)
        # 根据GPT-2论文，对残差投影应用特殊缩放初始化
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layers))
        # forward最后一次调用loss初始化，这将在使用目标张量调用forward时设置
        self.last_loss = None
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]
        
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)
        
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            # 给定目标值，计算损失
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            # 只转发最后一个位置输出
            logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim
            self.last_loss = None
        
        return logits
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """ 配置优化器参数
        """
        # 获取模型中所有的参数及名称
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 过滤出那些参数需要进行梯度计算
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        # 根据参数的维度创建优化器参数组
        # 具体规则是：任何二维及以上的参数（如权重矩阵和嵌入矩阵）都将应用权重衰减，
        # 其他一维参数（如偏置项和LayerNorm层的参数）则不应用权重衰减
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        # 构造优化器参数组列表，分别对应应用权重衰减和不应用权重衰减的参数
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # 计算并输出应用权重衰减和不应用权重衰减的参数数量
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # 创建AdamW优化器，并根据设备类型决定是否使用融合版本
        # 检查AdamW是否有'fused'参数，如果有且设备类型为CUDA，则使用融合版本
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        
        # 实例化AdamW优化器，传入优化器参数组列表以及学习率、beta参数等
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ 计算模型FLOPs利用率（MFU），单位为A100的bfloat16峰值FLOPS
        """
        # 首先估算每迭代一次模型所执行的FLOPs数量
        # 参考PaLM论文附录B：https://arxiv.org/abs/2204.02311
        # 统计模型所有参数的元素总数
        N = sum(p.numel() for p in self.parameters())
        # 获取模型参数配置
        cfg = self.args
        # 提取模型层数、头数、单个头的维度、最大序列长度
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        # 计算每步（token）所需FLOPs
        flops_per_token = 6*N + 12*L*H*Q*T
        # 计算每次前向和后向传播所需的FLOPs
        flops_per_fwdbwd = flops_per_token * T
        # 计算每迭代一次所需的FLOPs
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        # 计算每秒实际完成的FLOPs数量
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        # 计算模型FLOPs利用率（MFU）
        mfu = flops_achieved / flops_promised
        return mfu
    
    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """   
        根据给定的条件序列idx（形状为(b,t)的LongTensor）生成新序列，循环max_new_tokens次，
        每次都将上一轮的预测结果反馈回模型。注意，在此操作前通常需要确保模型处于eval模式。
        另外，此版本的采样方法未使用key/value缓存，效率较低。

        参数：
            - idx：条件序列索引，形状为(batch_size, 已有序列长度)
            - max_new_tokens：要生成的新toekn数量
            - temperature：控制采样分布平滑度的温度参数，默认为1.0
            - top_k：可选参数，限制采样的top-k个概率最大的选项

        返回值：
            - idx：最终生成的完整序列索引，形状为(batch_size, 已有序列长度 + 新增token数)
        """
        for _ in range(max_new_tokens):
            # 若条件序列过长，截取最近的max_seq_len个token作为条件子序列
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # 将条件子序列输入模型以获得序列中最后一个时间步的所有可能token的概率分布
            logits = self(idx_cond)
            # 截取仅包含最后一个时间步的logits
            logits = logits[:, -1, :] # crop to just the final time step
            if temperature == 0.0:
                # 当temperature为0时，选取概率最高的单一索引
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # 将logits除以温度参数以调整概率分布
                logits = logits / temperature
                # 如果指定了top_k参数，保留top_k个概率最高的选项
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # 将logits转换为归一化概率分布
                probs = F.softmax(logits, dim=-1)
                # 从概率分布中随机抽样一个索引
                idx_next = torch.multinomial(probs, num_samples=1)
            # 将抽样的新索引追加到当前条件序列末尾
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

def print_model_parameters(model):
    """ 打印模型各个层参数
    """
    param_sum = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_sum += param.numel()
            print(f"Layer: {name}, Parameters: {param.numel()}")
    print(f"Total of parameters: {param_sum}")    
    
if __name__ == "__main__":
    
    args = ModelArgs(
        dim=288,
        n_layers=6,
        n_heads=6,
        n_kv_heads=6,
        multiple_of=32,
        dropout=0.0,
        vocab_size=4096
    )
    model = TinyLlama2(args)
    
    # forward
    x = torch.tensor([[1,2,4],[4,3,2]])
    print(x.shape)
    y = model(x)
    # print(y)
    
    print_model_parameters(model)





