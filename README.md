# tiny-llama2-zh

## 1.简介

本项目旨在构建一个小参数量的中文Llama2大语言模型，包含：预训练 -> SFT指令微调 -> RLHF -> 量化。







## 2.快速开始



## 3.模型

### 3.1 Tokenizer

LLM分词器的构建方式有两种：一种是自己构造词表，训练一个分词器；另一种是选择开源模型训练好的分词器。

由于Llama2官方提供的词表中，中文部分较少。本项目为了方便，选择 [ChatGLM3](https://huggingface.co/THUDM/chatglm3-6b) 的分词器，该词表大小为64793。

### 3.2 模型尺寸

模型采用Llama2架构模型，未有任何改动，只是将参数缩放，具体参数细节如下所示：

| model            | dim  | n_layers | n_heads | n_kv_heads | max context length | params | vocab size |
| ---------------- | ---- | -------- | ------- | ---------- | ------------------ | ------ | ---------- |
| tiny-llama2-9m   | 64   | 5        | 8       | 4          | 512                | 9M     | 64793      |
| tiny-llama2-24m  | 288  | 6        | 6       | 6          | 512                | 24M    | 64793      |
| tiny-llama2-58m  | 512  | 8        | 8       | 8          | 512                | 58M    | 64793      |
| tiny-llama2-134m | 768  | 12       | 12      | 12         | 512                | 134M   | 64793      |
| tiny-llama2-256m | 1024 | 16       | 16      | 16         | 1024               | 256M   | 64793      |





## 4.预训练

### 4.1 网络小说模型训练

收集网络小说约9000本，经过清洗，去重，大约剩余37G文本左右，使用 [ChatGLM3](https://huggingface.co/THUDM/chatglm3-6b)  Tokenizer后，大约有 `9B` 的Token，具体网络小说数据集已上传 Hugging Face （[wdndev/webnovel-chinese · Datasets at Hugging Face](https://huggingface.co/datasets/wdndev/webnovel-chinese)）。

使用网络小说数据集，训练`tiny-llama2-24m`和`tiny-llama2-58m`这两个规格的模型，训练脚本 。。

#### （1）训练预料构建



#### （2）预训练脚本



#### （3）Loss曲线

训练Loss曲线如下所示：



#### （4）模型续写效果

24M模型续写效果



58M模型续写效果



### 4.2 通用模型训练

#### （1）预训练预料

本次训练的预训练预料都来自[Hugging Face](https://huggingface.co/)，主要包含以下几个经典的中文数据集，大约有19B左右Token，详细数据集如下：

| 中文预训练语料    | 链接 | 描述                                            |
| ----------------- | ---- | ----------------------------------------------- |
| Wiki中文百科      |      | 中文Wikipedia的数据                             |
| BaiduBaiKe        |      | 中文BaiduBaiKe的数据                            |
| zhihu             |      | 知乎KOL中截取的数据                             |
| TargetBot部分数据 |      | TargetBot模型训练的部分中文数据，原始数据太多了 |
|                   |      |                                                 |

上述数据处理脚本为，在处理时，Tokenizer后保存为可直接训练的二进制文件(`.bin`)。

注意：此处使用 Numpy 的格式保存，不需要考虑每个 max_seq_len 的长度，使用Numpy保存尽可能压缩存储空间。后续的SFT执行微调数据和RM数据集是以哦那个 pickle 格式保存，主要 Numpy 不能保存不等长列表。

#### （2）预训练预料构建





#### （3）预训练脚本





#### （4）Loss曲线





#### （5）预训练 Base 模型续写效果

##### Pytorch方式加载





##### Hugging Face方式加载







## 5.SFT指令微调

SFT指令微调在通用预训练模型上进行，即在 `base`模型上进行。

### 5.1 SFT指令微调预料



### 5.2 SFT指令微调预料构建



### 5.3 SFT指令微调脚本



### 5.4 SFT指令微调Loss曲线





### 5.5  微调 SFT 模型对话效果



#### （1）Pytorch方式加载





#### （2）Hugging Face方式加载









## 6.RLHF

### 6.1 RM模型





### 6.2 RL模型





## 7.DPO





## 8.量化



## 9.鸣谢

感谢下面这些开源项目，本项目实现有不少地方参考各个项目。

- [karpathy/llama2.c: Inference Llama 2 in one file of pure C ](https://github.com/karpathy/llama2.c)
- [DLLXW/baby-llama2-chinese](https://github.com/DLLXW/baby-llama2-chinese)
- [AI-Study-Han/Mini-Llama2-Chinese](https://github.com/AI-Study-Han/Mini-Llama2-Chinese)
- [Tongjilibo/build_MiniLLM_from_scratch](https://github.com/Tongjilibo/build_MiniLLM_from_scratch?tab=readme-ov-file)

