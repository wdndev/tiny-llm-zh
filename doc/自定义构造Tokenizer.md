# 自定义构造Tokenizer

## 1.简介

目前大模型的词表和分词器都是基于SentencePiece工具实现的，比如LLaMa，BLOOM，ChatGLM，Baichuan等，简单来说SentencePiece就是工程化的实现了之前写的各种的分词算法；而且实现的十分优雅，简单，快速，轻量。

使用SentencePiece的除了从0开始训练大模型的土豪和大公司外，大部分应该都是使用其为当前开源的大模型扩充词表，比如为LLaMa扩充通用中文词表（垂直领域词表），为开源的中文大模型ChatGLM，Baichuan扩充垂直领域词表。那这部分工作有没有意义呢？或者说值不值得投入资源去做呢？先说自己的结论，有，以下两点的作用：

1. **提高模型的编解码的效率**：在LLaMa原来的词表上，一个汉字平均1.45个token，扩充后的Chinese-LLaMa为0.65个token；那在垂直领域内呢？比如在LLaMa在继续扩充领域内词表，金融或者医疗等等，把“负债表”，“糖尿病”等领域词汇也加入词表里，那更加能提高其编解码的效率。
2. **提高模型的上下文窗口长度**：原LLaMa上下文长度是4096个token，不扩充词表前，按1.45来算就是最多只能输入2824个汉字，扩充后以0.65来算的话就是6301，垂直领域会更大。这点带来的好处是实打实的。

## 2.安装

SentencePiece的安装方式有两种，实现的效果是一样的。

### **2.1 源码安装**

在linux Ubuntu 上，命令行执行：

```bash
sudo apt-get update
sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev
```

然后下载源码进行安装：

```bash
git clone https://github.com/google/sentencepiece.git 
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
# linux
sudo ldconfig -v
# OSX/macOS
sudo update_dyld_shared_cache
```

验证安装是否成功：

```bash
spm_train --help
```

### **2.2 python安装**

通过pip安装，应该是最简单和快速的方式，实现的功能和上述源码安装是一模一样的：

```bash
pip install sentencepiece
```

## 3.使用

安装完成后，开始使用，第一步是训练词表，用起来很简单，源码安装的方式直接命令行执行

```bash
spm_train --input=<input> \
          --model_prefix=<model_name> \
          --vocab_size=8000 \
          --character_coverage=1.0 \
          --model_type=<type>
```

python的方式则是：

```python
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input=<input>, 
    model_prefix=<model_name>, 
    vocab_size=8000, 
    character_coverage=1.0, 
    model_type=<type>)
```

## 4.训练参数详解

以下是官方给出的训练参数解释，后面是通过实验的一些理解。

```bash
   --input (comma separated list of input sentences)  type: std::string default: ""
   --input_format (Input format. Supported format is `text` or `tsv`.)  type: std::string default: ""
   --model_prefix (output model prefix)  type: std::string default: ""
   --model_type (model algorithm: unigram, bpe, word or char)  type: std::string default: "unigram"
   --vocab_size (vocabulary size)  type: int32 default: 8000
   --accept_language (comma-separated list of languages this model can accept)  type: std::string default: ""
   --self_test_sample_size (the size of self test samples)  type: int32 default: 0
   --character_coverage (character coverage to determine the minimum symbols)  type: double default: 0.9995
   --input_sentence_size (maximum size of sentences the trainer loads)  type: std::uint64_t default: 0
   --shuffle_input_sentence (Randomly sample input sentences in advance. Valid when --input_sentence_size > 0)  type: bool default: true
   --seed_sentencepiece_size (the size of seed sentencepieces)  type: int32 default: 1000000
   --shrinking_factor (Keeps top shrinking_factor pieces with respect to the loss)  type: double default: 0.75
   --num_threads (number of threads for training)  type: int32 default: 16
   --num_sub_iterations (number of EM sub-iterations)  type: int32 default: 2
   --max_sentencepiece_length (maximum length of sentence piece)  type: int32 default: 16
   --max_sentence_length (maximum length of sentence in byte)  type: int32 default: 4192
   --split_by_unicode_script (use Unicode script to split sentence pieces)  type: bool default: true
   --split_by_number (split tokens by numbers (0-9))  type: bool default: true
   --split_by_whitespace (use a white space to split sentence pieces)  type: bool default: true
   --split_digits (split all digits (0-9) into separate pieces)  type: bool default: false
   --treat_whitespace_as_suffix (treat whitespace marker as suffix instead of prefix.)  type: bool default: false
   --allow_whitespace_only_pieces (allow pieces that only contain (consecutive) whitespace tokens)  type: bool default: false
   --control_symbols (comma separated list of control symbols)  type: std::string default: ""
   --control_symbols_file (load control_symbols from file.)  type: std::string default: ""
   --user_defined_symbols (comma separated list of user defined symbols)  type: std::string default: ""
   --user_defined_symbols_file (load user_defined_symbols from file.)  type: std::string default: ""
   --required_chars (UTF8 characters in this flag are always used in the character set regardless of --character_coverage)  type: std::string default: ""
   --required_chars_file (load required_chars from file.)  type: std::string default: ""
   --byte_fallback (decompose unknown pieces into UTF-8 byte pieces)  type: bool default: false
   --vocabulary_output_piece_score (Define score in vocab file)  type: bool default: true
   --normalization_rule_name (Normalization rule name. Choose from nfkc or identity)  type: std::string default: "nmt_nfkc"
   --normalization_rule_tsv (Normalization rule TSV file. )  type: std::string default: ""
   --denormalization_rule_tsv (Denormalization rule TSV file.)  type: std::string default: ""
   --add_dummy_prefix (Add dummy whitespace at the beginning of text)  type: bool default: true
   --remove_extra_whitespaces (Removes leading, trailing, and duplicate internal whitespace)  type: bool default: true
   --hard_vocab_limit (If set to false, --vocab_size is considered as a soft limit.)  type: bool default: true
   --use_all_vocab (If set to true, use all tokens as vocab. Valid for word/char models.)  type: bool default: false
   --unk_id (Override UNK (<unk>) id.)  type: int32 default: 0
   --bos_id (Override BOS (<s>) id. Set -1 to disable BOS.)  type: int32 default: 1
   --eos_id (Override EOS (</s>) id. Set -1 to disable EOS.)  type: int32 default: 2
   --pad_id (Override PAD (<pad>) id. Set -1 to disable PAD.)  type: int32 default: -1
   --unk_piece (Override UNK (<unk>) piece.)  type: std::string default: "<unk>"
   --bos_piece (Override BOS (<s>) piece.)  type: std::string default: "<s>"
   --eos_piece (Override EOS (</s>) piece.)  type: std::string default: "</s>"
   --pad_piece (Override PAD (<pad>) piece.)  type: std::string default: "<pad>"
   --unk_surface (Dummy surface string for <unk>. In decoding <unk> is decoded to `unk_surface`.)  type: std::string default: " ⁇ "
   --train_extremely_large_corpus (Increase bit depth for unigram tokenization.)  type: bool default: false
   --random_seed (Seed value for random generator.)  type: uint32 default: 4294967295
   --enable_differential_privacy (Whether to add DP while training. Currently supported only by UNIGRAM model.)  type: bool default: false
   --differential_privacy_noise_level (Amount of noise to add for DP)  type: float default: 0
   --differential_privacy_clipping_threshold (Threshold for clipping the counts for DP)  type: std::uint64_t default: 0
   --help (show help)  type: bool default: false
   --version (show version)  type: bool default: false
   --minloglevel (Messages logged at a lower level than this don't actually get logged anywhere)  type: int default: 0
```

#### （1）`input`&#x20;

指定训练语料文件，支持两种格式 `.txt` 和 `.tsv`（以制表符（Tab）作为分隔符的文件，类似于`.csv`文件），也可以传递以逗号分隔的文件列表。`.txt`文件内格式为每一行作为一个句子（sentences）。默认为`""`

```python
--input "/path/botchan.txt"
--input ["/path/botchan1.txt", "path/botchan2.txt"]
```

一般大规模训练时，会有几十个文件，在一个文件夹下，这时候可以通过sh脚本：

```python
files="/path/train_vocab/*" # 你的训练文件夹地址
file_list=$(echo $files | tr ' ' ',')
nohup spm_train --input $file_list
#...其他参数
```

#### （2）`input_format`

指定输入文件的格式，支持的格式有两种：`text`对应`.txt`；`tsv`对应`.tsv`。默认为`""`

#### （3）`model_prefix`

指定模型的输出前缀名，模型训练完成后，将使用这个前缀名来保存模型和词表文件。默认为`""`

#### （4）`model_type`

指定模型的分词算法，支持的选项有 `unigram`、`bpe`、`word`和`char`。默认为`"unigram"`

#### （5）`vocab_size`

指定词表大小，默认为8000

#### （6）`accept_language`

指定模型所支持的语言列表，多个语言可以用逗号分隔，语言代码是 ISO 639 标准定义的缩写，这个参数就是帮助模型识别语言，不设置也是可以的，默认为`""`

```python
--accept_language "en,zh"
```

#### （7）`character_coverage`

指定模型的字符覆盖率，较高的覆盖率可以使模型包含更多字符。**对于字符集丰富的语言（如日语或中文）推荐的默认值为 0.9995**，对于其他字符集较小的语言推荐默认值为 1.0。默认值为0.9995，如果词表比较大，或者说扩充的词表比较大，可以适当调大该参数。

#### （8）`input_sentence_size`

指定训练过程中加载的训练句子的最大数量。如果设置为非0值，模型将只加载小于设定的训练句子数量，默认为0，不设置数量。

#### （9）`shuffle_input_sentence`

当 `--input_sentence_size` 设置大于 0 时，此参数控制是否在加载输入句子之前对其进行随机采样，因为一般设置`input_sentence_size`时，是因为输入的句子太多了，比如输入的文件有1000个训练句子，但是只想要100个句子参与训练，这个时候设置这个参数就会随机采样100句。默认为True（但是`input_sentence_size` 设置大于 0 时才生效）。

#### （10）`seed_sentencepiece_size`

指定用于种子子词单元的最大数量，默认为1000000

#### （11）`num_threads`

指定在训练过程中使用的线程数，默认为16。

这个要在解释一下，这个线程只有在 EM-step阶段使用即这个参数`num_sub_iterations`，其他阶段都是单线程。原作者的回复：“Muti-thread computation is used only in the EM-step, after the seed vocab generation phase with suffix array.”所以大部分时间你只能看到只有一个CPU达到了100%，其他CPU都没有利用，作者说会在将来实现。。

#### （12）`max_sentencepiece_length`

指定子词单元的最大长度，默认为16。

#### （13）`max_sentence_length`

指定输入句子的最大长度，是以字节为单位的，默认为4192，UTF-8中一个汉字3个字节，大概就是`.txt`一行最多1397个汉字。

#### （14）`split_by_unicode_script`

指定是否用unicode脚本信息来划分子词单元，默认为True，解释一下Unicode 脚本，是 Unicode 标准中定义的一组字符集合，每个字符都被分配到一个或多个脚本（例如拉丁字母、希腊字母、汉字等）。当此参数启用时，模型在分割句子片段时会考虑每个字符所属的 Unicode 脚本，以便更好地处理不同脚本之间的边界，在多语言环境中非常有用。

#### （15）`split_by_number`

指定是否用数字来划分子词单元，默认为False，就是划分子词的时候要不要用数字来划分。

#### （16）`split_by_whitespace`

指定是否用空格来划分子词单元，默认为True

#### （17）`split_digits`

指定是否将所有数字字符拆分为单独的单元，就是将”2023“拆成”2“，”0“，”2“，”3“这种独立的子词单元，好处是减少词表的数字量，所有数字都能表示，坏处是token数量会变多，一个”2023“就是4个token，**默认是False，LLaMa是True**

#### （18）`treat_whitespace_as_suffix`

指定是否将空格字符作为子词的后缀而不是前缀，这里需要说明一下，空格也是基本字符，SentencePiece 使用元符号 "▁" (U+2581) 转义空格，这里的意思是空格放在前缀还是后缀，默认False

```python
"say hi"
前缀：["say","_hi"]
后缀：["say_","hi"]
```

#### （19）`allow_whitespace_only_pieces`

指定是否允许空格作为子词单元，就是单独的一个空格，默认为False。

#### （20）`control_symbols`

指定一组控制符号，这些符号用于划分子词，方便词汇表的构建，默认为""。

#### （21）`control_symbols_file`

指定包含一组控制符号的文件，这些符号用于划分子词，方便词汇表的构建，默认为""。

#### （22）`user_defined_symbols`

用户可以定义一组符号，这些符号可能不在训练文本中出现，但需要用于划分子词，默认为""。

#### （23）`required_chars`

指定一组 UTF-8 字符，它们将始终包含在生成的词汇表中，无论 `--character_coverage`参数的设置是多少，因为默认0.9995，并不会覆盖完全。

#### （24）`byte_fallback`

这个参数是比较重要的，用于**指定在遇到未知或很少的字符时将其分解为 UTF-8 字节来表示，这个参数打开了BPE实现的效果就和BBPE是一样的了**，比如”魑魅魍魉“，假如在训练语料中出现的次数太少，最后的词表里没有这个词，如果不开启这个参数就会OOV，如果开启了，这个词就会被用UTF-8的编码来分词即：`0xE9 0xAD 0x91 0xE9 0xAD 0x85 0xE9 0xAD 0x8D 0xE9 0xAD 0x89`，就可以被分词，分成12个token。默认为False。

#### （25）`vocabulary_output_piece_score`

是否将词汇表中的每个子词给出一个分数，默认为True。

#### （26）`normalization_rule_name`

指定文本规范化的规则，可以选择 `"nfkc"` 或 `"identity"`。解释一下`nfkc`：是一种常见的文本规范化规则，它使用了 Unicode 规范化形式 NFKC (Normalization Form KC)。NFKC 规范化通过将字符进行规范化，去除字符的多种表示形式，来确保文本在比较和处理时保持一致。它会将一些特定的字符组合转换为等效的单一字符，例如将带有重音符号的字符转换为没有重音符号的字符，以便更容易进行搜索、排序和匹配。identity:不对文本进行任何规范化处理。如果选择这个规则，文本将按照原始输入的方式进行处理，不进行任何字符合并、替换或重排。默认为nfkc。

#### （27）`normalization_rule_tsv`

允许从文件中加载自定义的文本规范化规则，默认为`""`

#### （28）`add_dummy_prefix`

是否在文本的开头添加一个虚拟的空格标记，以帮助处理文本的开头，默认为True。

#### （29）`remove_extra_whitespaces`

是否删除文本中的多余空格，包括开头、结尾和连续的内部空格，默认为True。

#### （30）`hard_vocab_limit`

如果启用，`--vocab_size` 参数将被视为硬限制，词汇表的大小不会超过该值。如果禁用，词汇表大小可能会略微超过指定的值，默认为True。

#### （31）`use_all_vocab`

如果启用，将使用子词作为词汇表，而不考虑 `--vocab_size` 参数的设置，默认为False。

#### （32）特殊ID

接下来是一系列特殊id的解释就不做赘述了，`unk_id`，`bos_id`，`eos_id`，`pad_id`，`unk_piece`等等，在训练词表时，这些最好和原要扩充的词表相对应。

#### （33）`train_extremely_large_corpus`

如果启用，将增加 unigram 分词算法中的比特深度，用于处理极大的训练语料库，只有在选用unigram 才生效，默认为False。

#### （34）`random_seed`

随机种子，如果随机种子是一样的，训练结果是可以重复的

#### （35）`enable_differential_privacy`

控制是否在训练过程中添加差分隐私设置，差分隐私：差分隐私用于防止模型过度依赖特定训练样本的信息，从而减少了对个体数据的敏感性。它通过在训练数据中引入噪音来实现这一点，使得模型不太可能准确地学习任何个别样本的细节信息。这有助于保护数据隐私，尤其是对于包含敏感信息的数据。仅在unigram 分词算法可设置。

## 5.扩充LLaMa 中文词表

### 5.1 训练中文分词

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

```python
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

执行上述训练过程，会在生成两个文件，chinese\_spm\_20000.model和chinese\_spm\_20000.vocab。看一下模型的分词效果：

```python
def test_chinese_spm(spm_model_path):
    sp_bpe = spm.SentencePieceProcessor() 
    sp_bpe.load(spm_model_path)
    print('*** BPE ***')
    print(sp_bpe.encode_as_pieces('翻译下面的句子为英文：有朋自远方来，不亦乐乎'))
    print(len(sp_bpe.encode_as_pieces('翻译下面的句子为英文：有朋自远方来，不亦乐乎')))

```

结果输出

```bash
*** BPE ***
['▁', '翻译', '下', '面的', '句', '子', '为', '英文', '：', '有', '朋', '自', '远', '方', '来', '，', '不', '亦', '乐', '乎']
20
```

### 5.2 合并LLaMa词表

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
    output_hf_dir = "tinyllm_tokenizer_hf_me"

    merge_tokenizer(llama_tokenizer_dir, chinese_sp_model_file, output_hf_dir)

    test_tokenizer(output_hf_dir)
```

至此，完成了LLaMa中文词表的扩充，扩充垂直领域词表也是如此，要准备垂直领域的训练语料，最好和通用领域的训练语料混合一下。

## 6.选读：扩充词表后的Embedding初始化

扩充词表后，该如何将新增的token再模型的embedding层和lm\_head层初始化？

以 llama2-7b为例

### 6.1 模型结构

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model_name = "/path/llama-2-7b-hf" # 模型的位置
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 新的分词器
new_tokenizer = AutoTokenizer.from_pretrained("/path/to/merged_tokenizer_hf") # 保存分词器的位置
```

加载完模型和分词器以及新增的分词器后，打印模型的结构：

```bash
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096, padding_idx=0)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
```

原来LLaMa的词表是`32000`，所以Embedding层的大小为 `(32000, 4096)`，即词表里的每一个token对应一个`1*4096`的Embedding向量，假设扩充后的词表大小为`49514`，有两种方式初始化扩充的词表：**随机扩充**和**均值扩充**

### 6.2 随即扩充

将扩充的token对应的向量随机初始化，实现方式如下：

```python
# 获取原先的embedding
embeddings = model.get_input_embeddings()
print(embeddings)
print(embeddings(torch.LongTensor([31999])))

```

```bash
# 结果
Embedding(32000, 4096, padding_idx=0)
tensor([[-8.3008e-03, -4.0588e-03, -1.1063e-03,  ...,  3.4790e-03,
         -1.2939e-02,  3.1948e-05]], device='cuda:0', dtype=torch.float16,
       grad_fn=<EmbeddingBackward0>)
```

resize词表

```python
model.resize_token_embeddings(49514)
new_embeddings = model.get_input_embeddings()
print(new_embeddings)
print(new_embeddings(torch.LongTensor([31999])))
```

```bash
# 结果
Embedding(49514, 4096)
tensor([[-8.3008e-03, -4.0588e-03, -1.1063e-03,  ...,  3.4790e-03,
         -1.2939e-02,  3.1948e-05]], device='cuda:0', dtype=torch.float16,
       grad_fn=<EmbeddingBackward0>)
```

可以看到，Embedding层从32000扩展为了49514，而且前32000个token的Embedding是没有发生变化的，只有新增的token是随机初始化的。

### 6.3 均值扩充

**新增token的Embedding用原来token的Embedding的均值来表示**，比如比如“你好”在原来的词表里为“你”：`[-0.0396, -0.0217, -0.0092, ..., -0.0032, -0.0103, 0.0068]`；“好”：`[-0.0104, -0.0145, -0.0142, ..., -0.0048, 0.0042, -0.0014]`，则新增的“你好”则为其均值：`[-0.0250, -0.0181, -0.0117, ..., -0.0040, -0.0030, 0.0027]`，以此方式扩充：

```python
# 新增的token和在原来token相对应的字典
token_mapping = {}
for i in range(32000, len(new_tokenizer)):
    # 使用 tokenizer 的 convert_ids_to_tokens 方法将索引转换为对应的 token
    token = new_tokenizer.convert_ids_to_tokens(i)
    # 原来的token
    input_ids = tokenizer(token, return_tensors="pt").input_ids[0]
    if input_ids[1] == 29871:
        new_input_ids = input_ids[2:]
    else:
        new_input_ids = input_ids[1:]        
    token_mapping[i] = new_input_ids

# 原始输入embedding
embeddings = model.get_input_embeddings()
# 新完全初始化的embedding
new_vocab_size = len(new_tokenizer)
embedding_dim = 4096
new_embedding = torch.nn.Embedding(new_vocab_size, embedding_dim)

# 将现有Embedding层的权重赋值给新的Embedding层的前32000行
num_to_copy = min(new_vocab_size, len(embeddings.weight))
new_embedding.weight.data[:num_to_copy, :] = embeddings.weight.data[:num_to_copy, :]

# 开始新增
for new_token, original_tokens in token_mapping.items():
    original_embeddings = embeddings(original_tokens)
    mean_embedding = torch.mean(original_embeddings, dim=0)
    new_embedding.weight.data[new_token] = mean_embedding

# 更换嵌入层
model.set_input_embeddings(new_embedding)
```

### 6.4 lm\_head扩充

同理模型的最后一层`lm_head`: `Linear(in_features=4096, out_features=32000, bias=False)`，参数也是一个`32000*4096`的矩阵，方法和上述是一致的，来看看均值扩充的方式：

```python
output_size = 32000
new_output_size = 49514
lm_head = model.lm_head
# 新的lm_head
new_lm_head = torch.nn.Linear(in_features=4096, out_features=new_output_size, bias=False)
# 前32000个向量不变
new_lm_head.weight.data[:output_size, :] = lm_head.weight.data[:output_size, :]

# 新增
for new_token, original_tokens in token_mapping.items():
    original = 0
    for i in original_tokens:
        original += lm_head.weight.data[i]
    mean_para = original / len(original_tokens)
    new_lm_head.weight.data[new_token] = mean_para

# 替换模型原来的lm_head
model.lm_head = new_lm_head
# 修改词表大小
model.config.vocab_size = 49514
# 最后完成了embedding和lm_head替换后，保存模型
model.save_pretrained("llama-2-7b-extent", max_shard_size="8GB")
```

### 6.5 总结

扩充词表后，需要对模型的`embedding`和`lm_head`做的操作，目前**业界一般用的都是均值的方式**。

可以算一下新增了多少个参数：`(40114-32000) * 4096 * 2 = 66,469,888`，6千多万个参数，还只是扩充8千个词，如果扩充的词表数量达到5万左右，那新增参数就是1亿多，这个参数数量还是不少的。
