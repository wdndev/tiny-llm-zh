## Tiny LLM Tokenizer

## 1.简介

采用扩充 LLaMA2 词表的方式构建 Tiny LLM 词表。

由于原版 LLaMA2 对中文的支持非常有限，本项目在原版 LLaMA 的基础上进一步扩充了中文词表。

在通用中文语料上训练了基于 sentencepiece 的 20K 中文词表并与原版LLaMA模型的 32K 词表进行合并，排除重复的token后，并添加特殊 token 后，最终得到的最终中文LLaMA词表大小为 49958

注意：预训练用的是ChatGLM3的词表，并未使用扩充的词表

## 2.词表扩种

### 2.1 训练中文分词

准备一份中文训练语料保存为按照每一行保存为 `.txt`文件，选用百科的所有语料，大约8G左右语料，存储为txt文本，其中划分句子代码如下：

```python
def split_sentences(text):
    """
    分割文本为句子列表
    """
    # 正则表达式匹配中英文句子结尾标点
    endings_pattern = r'(?<![.?!。？！])[.?!。？！]'
    # 匹配所有的句子结尾标点位置
    sentence_end_positions = [m.end() for m in re.finditer(endings_pattern, text)]
    
    # 添加文本末尾位置以确保处理最后一个句子
    if text and text[-1] not in ".?!。？！":
        sentence_end_positions.append(len(text))
    
    # 分割句子
    sentences = [text[start:end] for start, end in zip([0] + sentence_end_positions[:-1], sentence_end_positions)]
    
    return sentences
```

开始训练，这里面有几个参数要注意一下：

- 词表大小为 20k
- `model_type`分词算法选择`bpe`，
- `split_digits`为True，`byte_fallback`为True，和LLaMa 保持一致，
- `max_sentence_length`设置的大一点

```Python
import sentencepiece as spm
import os
import glob

def tain_chinses_spm(input_txt_dir, vocab_size, output_dir="."):
    # 保存的模型名称
    prefix = os.path.join(output_dir, f"chinese_spm_{vocab_size}")

    text_filenames = sorted(glob.glob(os.path.join(input_txt_dir, "*.txt")))
    print("file list: ", text_filenames)

    # 2) train the sentencepiece model
    print("Will now train the vocab...")
    spm.SentencePieceTrainer.train(input=text_filenames,
                                   model_prefix=prefix,
                                   model_type="bpe",
                                   vocab_size=vocab_size,
                                   self_test_sample_size=0,
                                   input_format="text",
                                   character_coverage=0.9995,
                                   num_threads=os.cpu_count(),
                                   split_digits=True,       # 是否将数字划分为单个 token, 在 llama 中是这么做的
                                   allow_whitespace_only_pieces=True,
                                   byte_fallback=True,
                                   unk_surface=r" \342\201\207 ",
                                   max_sentence_length=24000)


    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")

if __name__ == "__main__":
    input_txt_dir = "baike_txt"
    vocab_size = 20000
    output_dir = "sp_output"
    tain_chinses_spm(input_txt_dir, vocab_size, output_dir)
```

执行上述训练过程，会在生成两个文件，chinese_spm_20000.model和chinese_spm_20000.vocab。看一下模型的分词效果：

```Python
def test_chinese_spm(spm_model_path):
    sp_bpe = spm.SentencePieceProcessor() 
    sp_bpe.load(spm_model_path)
    print('*** BPE ***')
    print(sp_bpe.encode_as_pieces('翻译下面的句子为英文：有朋自远方来，不亦乐乎'))
    print(len(sp_bpe.encode_as_pieces('翻译下面的句子为英文：有朋自远方来，不亦乐乎')))

```

结果输出

```Bash
*** BPE ***
['▁', '翻译', '下', '面的', '句', '子', '为', '英文', '：', '有', '朋', '自', '远', '方', '来', '，', '不', '亦', '乐', '乎']
20
```

### 2.2 合并LLaMa词表

参考代码见：

```python
import os
# 设置环境变量，指定protobuf的Python实现为纯Python版本
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import argparse
import json

def merge_tokenizer(llama_tokenizer_dir, chinese_sp_model_file, output_hf_dir="tinyllm_tokenizer_hf"):
    # 加载LlamaTokenizer
    llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
    # 中文sentencepiece模型
    chinese_sp_model = spm.SentencePieceProcessor()
    chinese_sp_model.Load(chinese_sp_model_file)

    # 将LlamaTokenizer加载为protobuf模型对象
    llama_spm = sp_pb2_model.ModelProto()
    llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
    # 将中文模型加载为protobuf模型对象
    chinese_spm = sp_pb2_model.ModelProto()
    chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())

    # 打印基本信息
    print("llama token nums: ", len(llama_tokenizer))
    print("chinese sp nums: ", len(chinese_sp_model))

    # 向 LLaMA 的 tokenizer 中添加中文 tokens
    ## 1.首先创建一个set包含所有LLaMA的tokens以加速查找
    llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)
    ## 2.遍历中文模型的tokens，如果不在 LLaMA 的 tokens 集合中，则添加至 LLaMA 模型
    for p in chinese_spm.pieces:
        piece = p.piece
        if piece not in llama_spm_tokens_set:
            # 创建新的SentencePiece对象
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece     # 设置token内容
            new_p.score = 0         # 设置默认的分数
            llama_spm.pieces.append(new_p)  # 添加到LLaMA的模型pieces中

    # 保存合并后的模型
    output_sp_dir = 'tmp_tinyllm_tokenizer_sp'  # 保存sentencepiece模型的目录
    os.makedirs(output_sp_dir, exist_ok=True)  # 确保目录存在
    # 保存sentencepiece模型到文件
    with open(output_sp_dir + '/tokenizer.model', 'wb') as f:
        f.write(llama_spm.SerializeToString())
    
    # 使用新生成的vocab文件初始化LlamaTokenizer，并保存为 Hugging Face 格式
    tokenizer = LlamaTokenizer(vocab_file = output_sp_dir + '/tokenizer.model', legacy=True)
    ## 添加特殊 token
    custom_special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|im_start|>", "<|im_end|>"]
    for token in custom_special_tokens:
        tokenizer.add_tokens(token)
    
    tokenizer.save_pretrained(output_hf_dir)
    print(f"tinyllm token num: {len(tokenizer)}")
    print(f"Tiny LLM tokenizer has been saved to {output_hf_dir}")

def test_tokenizer(hf_tokenizer_dir):
    tinyllm_tokenizer = LlamaTokenizer.from_pretrained(hf_tokenizer_dir)
    print("tinyllm tokenizer nums: ", len(tinyllm_tokenizer))

    sys_text = "你是由wdndev开发的个人助手。"
    user_text = "翻译下面的句子为英文：有朋自远方来，不亦乐乎"
    answer_text = "It is always a pleasure to greet a friend from afar."
    input_txt = "\n".join(["<|system|>", sys_text.strip(), 
                            "<|user|>", user_text.strip(), 
                            "<|assistant|>"]).strip() + "\n" + answer_text.strip()

    print("-----input text: \n", input_txt)

    encode_ids = tinyllm_tokenizer.encode(input_txt, add_special_tokens=False)
    print("-----encode ids: \n", encode_ids)

    decode_ids = tinyllm_tokenizer.decode(encode_ids)
    print("-----dencode ids: \n", decode_ids)


if __name__ == "__main__":

    llama_tokenizer_dir = "input_dir/llama2_tokenizer"
    chinese_sp_model_file = "sp_output/chinese_spm_20000.model"
    output_hf_dir = "tinyllm_tokenizer_hf"

    merge_tokenizer(llama_tokenizer_dir, chinese_sp_model_file, output_hf_dir)

    test_tokenizer(output_hf_dir)
```

至此，完成了LLaMa中文词表的扩充，扩充垂直领域词表也是如此，要准备垂直领域的训练语料，最好和通用领域的训练语料混合一下。

