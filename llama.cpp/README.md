# Tiny LLM llama.cpp

## 1.简介

Tiny LLM 92M 模型已支持 llama.cpp C++ 推理框架，建议在 linux 环境下测试，windows效果不好；

所支持 llama.cpp 为自己修改的版本，仓库链接为： [llama.cpp.tinyllm](https://github.com/wdndev/llama.cpp.tinyllm)

### 1.1 llama.cpp

llama.cpp 是一个C++库，用于简化LLM推理的设置。它使得在本地机器上运行Qwen成为可能。该库是一个纯C/C++实现，不依赖任何外部库，并且针对x86架构提供了AVX、AVX2和AVX512加速支持。此外，它还提供了2、3、4、5、6以及8位量化功能，以加快推理速度并减少内存占用。对于大于总VRAM容量的大规模模型，该库还支持CPU+GPU混合推理模式进行部分加速。本质上，llama.cpp的用途在于运行GGUF（由GPT生成的统一格式）模型。

### 1.2 gguf

GGUF是指一系列经过特定优化，能够在不同硬件上高效运行的大模型格式。这些模型格式包括但不限于原始格式、exl2、finetuned模型（如axolotl、unsloth等）。每种格式都有其特定的应用场景和优化目标，例如加速模型推理、减少模型大小、提高模型准确性等。


## 2.使用

### 2.1 准备

建议使用 linux 系统

```shell
git clone https://github.com/wdndev/llama.cpp.tinyllm
cd llama.cpp.tinyllm
```

然后运行 make 命令：

```shell
make
```

然后你就能使用 `llama.cpp` 运行GGUF文件。

### 2.2 模型转化

先需要按照如下所示的方式为fp16模型创建一个GGUF文件：

```shell
python convert-hf-to-gguf.py wdndev/tiny_llm_sft_92m --outfile models/tinyllm/tinyllm-92m-fp16.gguf
```

其中，第一个参数指代的是预训练模型所在的路径或者HF模型的名称，第二个参数则指的是想要生成的GGUF文件的路径；在运行命令之前，需要先创建这个目录。

下面需要根据实际需求将其量化至低比特位。以下是一个将模型量化至4位的具体示例：

```shell
./llama-quantize models/tinyllm/tinyllm-92m-fp16.gguf  models/tiny_llm_92m/tinyllm-92m-q4_0.gguf q4_0
```

到现在为止，已经完成了将模型量化为4比特，并将其放入GGUF文件中。这里的 q4_0 表示4比特量化。现在，这个量化后的模型可以直接通过llama.cpp运行。

### 2.3 推理

使用如下命令可以运行模型

```shell
./llama-cli -m ./models/tinyllm/tinyllm-92m-fp16.gguf -p "<|system|>\n你是由wdndev开发的个人助手。\n<|user|>\n请介绍一下北京，你好。\n<|assistant|>\n" -n 128 --repeat-penalty 1.2 --top-p 0.8 --top-k 0
```

`-n` 指的是要生成的最大token数量。这里还有其他超参数供你选择，并且你可以运行

```shell
./llama-cli -h
```
以了解它们


