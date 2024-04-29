# Tiny LLM MoE zh

## 1.简介

本项目旨在构建一个小参数量的中文语言大模型，用于快速入门学习大模型相关知识，如果此项目对你有用，可以点一下start，谢谢！

模型架构：整体模型架构采用开源通用架构，包括：RMSNorm，RoPE，MHA等

实现细节：实现大模型两阶段训练及后续人类对齐，即：预训练(PTM) -> 指令微调(SFT) -> 人类对齐(RLHF, DPO) -> 测评。

本项目主要有三个分支，推荐学习 主分支，具体区别如下：

- [llama2_torch](https://github.com/wdndev/tiny-llm-zh/tree/llama2_torch) ： 模型架构采用原版 Llama2 架构，只是将部分的输入输出修改为适合训练的格式；
- `main`   `tiny_llm` ： 对齐开源社区模型，使用Transformers库构建底层模型，也使用Transformers库进行多卡多机训练；
- [tiny_llm_moe](https://github.com/wdndev/tiny-llm-zh/tree/tiny_llm_moe) ： 在`tiny_llm`的基础上，修改 `MLP`层为MoE模型，使用Transformers库进行多卡多机训练。

注意：

1. 因资源限制，本项目的第一要务是走通大模型整个流程，而不是调教比较好的效果，故评测结果分数较低，部分生成结构错误。

## 2.快速开始

模型已托管在 Huggingface 和 ModeScope 中，可运行代码自动下载。

建议使用Huggingface 下载，如果Huggingface 下载失败，再使用 ModeScope 下载模型后，修改`model_id`中的路径为本地目录，即可运行。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "wdndev/hf_tiny_llm_58m_sft"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True) 
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)

# text = "介绍一下刘德华。"
# text = "请问，世界上最大的动物是什么？"
text = "中国的首都在什么地方？"

# 哎。。。，SFT时没有注意这个特殊的token，拼接了prompt和answer，使用HF时，词表中没有这个，，，难受
model_inputs_id = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.special_tokens['<bos>']]
model_inputs_id = (torch.tensor(model_inputs_id, dtype=torch.long, device=model.device)[None, ...])
generated_ids = model.generate(model_inputs_id)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs_id, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
```
生成效果
```bash
问：介绍一下刘德华。
答：刘德华是中国著名的演员、歌手、电影制片人、音乐制作人、演员和导演。他因创作的电影作品深受观众喜爱，并经常获得奥斯卡最佳男主角奖。

问：中国的首都在什么地方？
答：中国的首都在北京。

问：请问，世界上最大的动物是什么？
答：蓝鲸是世界上最大的动物。

```









