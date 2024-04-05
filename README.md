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

| model            | dim  | n_layers | n_heads | n_kv_heads | max context length | params | vocab size | loss | download |
| ---------------- | ---- | -------- | ------- | ---------- | ------------------ | ------ | ---------- | ---- | -------- |
| tiny-llama2-9m   | 64   | 5        | 8       | 4          | 512                | 9M     | 64793      |      |          |
| tiny-llama2-24m  | 288  | 6        | 6       | 6          | 512                | 24M    | 64793      |      |          |
| tiny-llama2-58m  | 512  | 8        | 8       | 8          | 512                | 58M    | 64793      |      |          |
| tiny-llama2-134m | 768  | 12       | 12      | 12         | 512                | 134M   | 64793      |      |          |
| tiny-llama2-256m | 1024 | 16       | 16      | 16         | 1024               | 256M   | 64793      |      |          |





## 4.预训练



## 5.SFT指令微调



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

