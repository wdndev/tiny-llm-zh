"""
Download, preprocess and serve the TinyStories dataset as a DataLoader.
"""

import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import requests
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the TinyStories dataset, unless it's already downloaded
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    with open(shard_filenames[0], "r") as f:
        data = json.load(f)
    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    print(f"Example story:\n{data[0]}")

def train_vocab(vocab_size):
    """
    在TinyStories数据集上训练自定义的SentencePiece分词器。
    自定义分词器文件将会保存在DATA_CACHE_DIR/tok{N}目录下,  
    其中N为词汇表大小。这里也会存放预处理过的.bin文件。
    """
    # 断言词汇表大小必须大于0
    assert vocab_size > 0, "Vocab size must be positive"

    # SentencePiece输出文件前缀路径
    prefix = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")

    # 设置用于词汇表训练的数据分片数量,  为了效率保持较低值
    num_shards = 10

    # 1) 将大量文本数据导出为单个文本文件tiny.txt
    tiny_file = os.path.join(DATA_CACHE_DIR, "tiny.txt")
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    print(f"Writing temporary file {tiny_file} with {num_shards} shards...")
    # 将前num_shards个数据分片的内容写入tiny.txt文件
    with open(tiny_file, "w", encoding="utf-8") as of:
        for shard in tqdm(shard_filenames[:num_shards]):
            with open(shard, "r") as f:
                data = json.load(f)
            for example in data:
                # 获取故事文本
                text = example["story"]
                # 移除首尾空格
                text = text.strip()
                # 写入文本行,  并在结尾添加换行符
                of.write(text + "\n")
    print(f"Size is: {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")

    # 2) 训练SentencePiece模型
    print("Will now train the vocab...")
    # 使用SentencePieceTrainer训练模型,  参数详细配置如下 : 
    # - 输入文件: tiny_file
    # - 模型前缀 : prefix
    # - 模型类型 : "bpe"（Byte Pair Encoding）
    # - 词汇表大小 : vocab_size
    # - 自我测试样本大小 : 0
    # - 输入格式 : "text"
    # - 字符覆盖率 : 1.0（确保所有字符都在词汇表中）
    # - 并发线程数 : os.cpu_count()（使用全部可用CPU核心）
    # - 是否允许数字拆分 : True
    # - 是否允许纯空白字符作为单独片段 : True
    # - 是否启用字节级回退 : True
    # - 未知字符（UNK）的表面形式 : "\342\201\207"（Unicode空心圆圈）
    # - 规范化规则名称 : "identity"（不做额外规范化）
    spm.SentencePieceTrainer.train(input=tiny_file,
                                   model_prefix=prefix,
                                   model_type="bpe",
                                   vocab_size=vocab_size,
                                   self_test_sample_size=0,
                                   input_format="text",
                                   character_coverage=1.0,
                                   num_threads=os.cpu_count(),
                                   split_digits=True,
                                   allow_whitespace_only_pieces=True,
                                   byte_fallback=True,
                                   unk_surface=r" \342\201\207 ",
                                   normalization_rule_name="identity")

    # 3) 可选清理步骤,  询问用户是否删除临时文件tiny.txt
    dec = input(f"Delete the temporary file {tiny_file}? [y/N] ")
    if dec.lower() == "y":
        os.remove(tiny_file)
        print(f"Deleted {tiny_file}")

    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")


def process_shard(args, vocab_size):
    """ 处理分片函数
    """
    shard_id, shard = args
    # 获取指定词汇表大小对应的分词器模型路径
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)
    # 打开并读取JSON格式的分片文件
    with open(shard, "r") as f:
        data = json.load(f)
    # 初始化存储所有 tokens 的列表
    all_tokens = []
    # 训练分片每一个例子
    for example in tqdm(data, position=shard_id):
        # 获取并去除故事文本的前后空白字符
        text = example["story"]
        text = text.strip()  # get rid of leading/trailing whitespace
        # 使用分词器对文本进行编码,  添加BOS标记但不添加EOS标记
        tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
        # 将编码后的token添加到列表中
        all_tokens.extend(tokens)
    # 将token列表转换为uint16类型的numpy数组
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    # 计算输出的二进制文件名
    if vocab_size == 0:
        # 如果使用的是Llama 2,  将编码后的文件保存在同一目录下,  替换.json为.bin
        tokenized_filename = shard.replace(".json", ".bin")
    else:
        # 为不同词汇表大小创建新的tok{N}目录,  并将.bin文件保存在此目录下
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)
    
    # 将numpy数组中的所有字节写入二进制文件
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    # 计算平均序列长度（序列之间由BOS标记1分隔）
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")


def pretokenize(vocab_size):
    """ 定义预处理函数
    """
    # 获取所有分片文件路径
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    
    # 如果词汇表大小大于0,  创建tok{N}目录以存储.bin文件
    if vocab_size > 0:
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)

    # 定义处理分片的函数,  并将其应用于参数
    fun = partial(process_shard, vocab_size=vocab_size)
    # 使用进程池执行器并行处理所有分片
    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))
    print("Done.")


class PretokDataset(torch.utils.data.IterableDataset):
    """ 加载磁盘上的预处理样本,  并以PyTorch张量的形式逐个产出
    """

    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        self.split = split
        # 每个样本的最大序列长度
        self.max_seq_len = max_seq_len
        # 词汇表大小
        self.vocab_size = vocab_size
        # 词汇来源（"llama2" 或 "custom"）
        self.vocab_source = vocab_source

    def __iter__(self):
        # 获取当前DataLoader中工作进程的信息
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # 获取分布式训练环境中的全局rank信息
        rank = dist.get_rank() if dist.is_initialized() else 0
        # 依据worker_id和rank生成一个唯一的随机种子
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        # 根据词汇来源确定.bin文件所在目录
        if self.vocab_source == "llama2":
            # .bin文件与.json文件同目录
            bin_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        elif self.vocab_source == "custom":
            # .bin文件在tok{N}目录下
            bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{self.vocab_size}")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        # 根据split参数选择训练集或测试集
        # 如果是train,  除去第一个分片作为测试集,  其余作为训练集；否则,  只使用第一个分片作为测试集
        shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        assert len(shard_filenames)>0, f"No bin files found in {bin_dir}"
        # 无限循环,  每次迭代都重新洗牌并处理所有分片
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # 使用memmap方式打开.bin文件,  使其保持在磁盘上,  而非完全加载到内存
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                # 计算能组成多少个完整批次
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # 去掉最后一个不足一个完整批次的部分
                assert num_batches > 0, "this shard is way too small? investigate."
                
                # 随机打乱批次顺序
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                
                # 遍历并产出每个批次
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # 将memmap数据复制到新的numpy数组中,  然后转换为int64类型并转换为PyTorch张量
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    # 划分输入序列x和目标序列y
                    x = chunk[:-1]
                    y = chunk[1:]
                    # 产出当前批次的(x, y)样本对
                    yield x, y

# -----------------------------------------------------------------------------
# public interface functions

def get_tokenizer_model_path(vocab_size):
    """
    Returns path to the sentencepiece tokenizer model for a given vocab size
    vocab_size = 0 designates the default Llama 2 tokenizer, in that case
    None is returned.
    """
    if vocab_size == 0:
        return None
    else:
        return os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}.model")

class Task:

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        """ 静态方法,  用于创建一个预处理数据迭代器,  将数据按批次生成,  并将每批次的数据移动至指定设备上。
        参数 : 
            - batch_size : 整数,  指定每个批次包含的数据样本数量。
            - device : torch.device对象,  指明数据应该被移动到的设备,  如CPU或GPU。
            - num_workers (默认值=0) : 整数,  设置用于数据加载的子进程数,  若大于0,  则启用多线程数据加载。
            - **dataset_kwargs : 关键字参数字典,  用于传递给PretokDataset类初始化时所需的其他参数。
        """
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            # 将批次数据转移到device设备上，非阻塞方式加速转移过程
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # 产出已转移至设备上的批次数据
            yield x, y

# -----------------------------------------------------------------------------
# CLI for constructing the dataset

if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    python tinystories.py download
    python tinystories.py pretokenize

    To tokenize data with a custom tokenizer we train ourselves with sentencepiece, e.g.:
    python tinystories.py download
    python tinystories.py train_vocab --vocab_size=2048
    python tinystories.py pretokenize --vocab_size=2048
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "pretokenize", "train_vocab"])
    parser.add_argument("--vocab_size", type=int, default=0, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "download":
        download()
    elif args.stage == "train_vocab":
        train_vocab(vocab_size=args.vocab_size)
    elif args.stage == "pretokenize":
        pretokenize(vocab_size=args.vocab_size)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
