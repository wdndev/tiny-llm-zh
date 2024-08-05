import os
import pickle
import jsonlines
import hashlib
import random
import pandas as pd
import numpy as np
from typing import Optional, Dict
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from utils.chatglm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
from datasets import load_dataset
import datasets


class PTMDataset(Dataset):
    def __init__(self, data_path_list, max_length=512, memmap=False):
        super(PTMDataset, self).__init__()
        #
        if memmap:
            with open(data_path_list[0],'r') as f:
                nbytes = f.seek(0,2)
                flen = f.tell() // np.dtype('uint16').itemsize
            self.data = np.memmap(data_path_list[0], dtype=np.dtype('uint16'), shape=(flen // max_length, max_length))
        else:
            data_list = []
            for data_path in data_path_list:
                with open(data_path,'rb') as f:
                    data = np.fromfile(f, dtype=np.uint16)
                    data_list.append(data)
            data = np.concatenate(data_list)
            data = data[:max_length * int(len(data) / max_length)]
            #np.random.shuffle(data)
            self.data = data.reshape(-1, max_length)

        self.shuffle_index = list(range(len(self.data)))
        random.shuffle(self.shuffle_index)
        print("memmap:{} train data.shape:{}".format(memmap, self.data.shape))
        print("Data loading is completed.")
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        index = self.shuffle_index[index]
        sample = self.data[index]
        X = np.array(sample).astype(np.int64)
        # Y = np.array(sample[1:]).astype(np.int64)
        input_ids = torch.LongTensor(X)
        # labels = torch.LongTensor(Y)

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }

class PTMDatasetMap(Dataset):
    """ 数据集类，使用内存映射处理大文件数据，以节省内存。
    """
    def __init__(self, data_path_list: list, max_length=512):
        super(PTMDatasetMap, self).__init__()
        
        self.data = []          # 存储每个文件的内存映射数据
        self.index_map = {}     # 索引映射，便于快速定位样本
        self.token_size = 0     # token数
        self.data_size = 0      # 样本数量

        for idx, file_path in enumerate(data_path_list):
            with open(file_path, 'r') as f:
                nbytes = f.seek(0, 2)
                flen = f.tell() // np.dtype('uint16').itemsize
            
            # 更新统计信息和索引映射
            self.token_size += flen
            self.index_map.update({self.data_size + i : (idx, i) for i in range(flen // max_length)})
            self.data_size += flen // max_length

            # 使用内存映射读取数据
            self.data.append(np.memmap(file_path, dtype=np.dtype('uint16'), shape=(flen // max_length, max_length)))

        print('total token size: {:,} token,  data sample size: [{:,}, {}]'.format(self.token_size, self.data_size, max_length))

        # 初始化时生成随机索引列表
        self.shuffled_indices = list(self.index_map.keys())
        random.shuffle(self.shuffled_indices)

        # print(self.shuffled_indices)
        
    def __len__(self):
        return self.data_size

    def __getitem__(self, index: int):
        real_index = self.shuffled_indices[index]
        fi, i = self.index_map[real_index]
        sample = self.data[fi][i]
        X = np.array(sample[:-1]).astype(np.int64)
        Y = np.array(sample[1:]).astype(np.int64)
        input_ids = torch.LongTensor(X)
        labels = torch.LongTensor(Y)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

class SFTDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        max_length=256,
        system: str = "你是由wdndev开发的个人助手。",
    ):
        super(SFTDataset, self).__init__()
        self.data = []
        with jsonlines.open(data_path) as reader:
            for obj in reader:
                self.data.append(obj)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system = system
        
        for ex in self.data[:5]:
            self.preprocessing(ex, debug=True)

        print("SFT Data loading is completed. data len: ", len(self.data))

    def preprocessing(self, example, debug=False):
        """
            [gMASK]sop <|system|>
            prompt_txt
            <|user|>
            user_txt
            <|assistant|>
            assistant_txt
        示例：
            [gMASK]sop <|system|>
            你是由wdndev开发的个人助手。
            <|user|>
            为以下地点提供纬度和经度：- 纽约市
            - 巴黎
            - 北京
            <|assistant|>
            - 纽约市：纬度：40.7128°N，经度：74.0060°W
            - 巴黎：纬度：48.8566°N，经度：2.3522°E
            - 北京：纬度：39.9042°N，经度：116.4074°E
        """
        input_ids, labels = [], []
        prompt_txt = self.system
        # print(type(example))
        user_txt = example["question"]
        assistant_txt = example["answer"]

        instruction = self.tokenizer.encode(text="\n".join(["<|system|>", prompt_txt.strip(), 
                                    "<|user|>", user_txt.strip(), 
                                    "<|assistant|>"]).strip() + "\n",
                                    add_special_tokens=True, 
                                    truncation=True, 
                                    max_length=self.max_length)
        response = self.tokenizer.encode(assistant_txt.strip(), add_special_tokens=False, truncation=True, max_length=self.max_length)
        input_ids = instruction + response + [self.tokenizer.eos_token_id]
        labels = [self.tokenizer.pad_token_id] * len(instruction) + response + [self.tokenizer.eos_token_id]
        if (len(input_ids) > self.max_length):
            return None
        if debug:
            print(self.tokenizer.decode(input_ids))
            print("-------------------------------")
        
        pad_len = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        labels += [self.tokenizer.pad_token_id] * pad_len
        labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        processed_example = self.preprocessing(self.data[idx])
        while processed_example is None:
            idx = (idx + 1) % len(self.data)  # 循环至下一个有效样本
            processed_example = self.preprocessing(self.data[idx])

        return processed_example

class RMDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        max_length=256,
        system: str = "你是由wdndev开发的个人助手。",
    ):
        super(RMDataset, self).__init__()
        self.data = []
        with jsonlines.open(data_path) as reader:
            for obj in reader:
                self.data.append(obj)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system = system
        
        for ex in self.data[:5]:
            self.preprocessing(ex, debug=True)
        
        print("RM Data loading is completed. data len: ", len(self.data))

    def preprocessing(self, example, debug=False):
        """
        示例：
            [gMASK]sop <|system|>
            你是由wdndev开发的个人助手。
            <|user|>
            为以下地点提供纬度和经度：- 纽约市
            - 巴黎
            - 北京
            <|assistant|>
            - 纽约市：纬度：40.7128°N，经度：74.0060°W
            - 巴黎：纬度：48.8566°N，经度：2.3522°E
            - 北京：纬度：39.9042°N，经度：116.4074°E
        """
        input_ids, labels = [], []
        prompt_txt = self.system
        # print(type(example))
        user_txt = example["prompt"]
        assistant_chosen_txt = example["chosen"]
        assistant_rejected_txt = example["rejected"]

        instruction = self.tokenizer.encode(text="\n".join(["<|system|>", prompt_txt.strip(), 
                                    "<|user|>", user_txt.strip(), 
                                    "<|assistant|>"]).strip() + "\n",
                                    add_special_tokens=True, 
                                    truncation=True, 
                                    max_length=self.max_length)

        response_j = self.tokenizer.encode(assistant_chosen_txt.strip(), add_special_tokens=False, truncation=True, max_length=self.max_length)
        input_ids_j = instruction + response_j + [self.tokenizer.eos_token_id]
        pad_len_j = self.max_length - len(input_ids_j)
        input_ids_j += [self.tokenizer.pad_token_id] * pad_len_j

        response_k = self.tokenizer.encode(assistant_rejected_txt.strip(), add_special_tokens=False, truncation=True, max_length=self.max_length)
        input_ids_k = instruction + response_k + [self.tokenizer.eos_token_id]
        pad_len_k = self.max_length - len(input_ids_k)
        input_ids_k += [self.tokenizer.pad_token_id] * pad_len_k


        if debug:
            print(self.tokenizer.decode(input_ids_j))
            print("******************")
            print(self.tokenizer.decode(input_ids_k))
            print("-------------------------------")

        if (len(input_ids_j) > self.max_length or len(input_ids_k) > self.max_length):
            return None

        input_ids_j = torch.LongTensor(input_ids_j)
        attention_mask_j = input_ids_j.ne(self.tokenizer.pad_token_id)

        input_ids_k = torch.LongTensor(input_ids_k)
        attention_mask_k = input_ids_k.ne(self.tokenizer.pad_token_id)

        return {
            "input_ids_j": input_ids_j,
            "attention_mask_j": attention_mask_j,
            "input_ids_k": input_ids_k,
            "attention_mask_k": attention_mask_k,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # return self.preprocessing(self.data[idx])
        processed_example = self.preprocessing(self.data[idx])
        while processed_example is None:
            idx = (idx + 1) % len(self.data)  # 循环至下一个有效样本
            processed_example = self.preprocessing(self.data[idx])

        return processed_example

class RLDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        max_length=256,
        system: str = "你是由wdndev开发的个人助手。",
    ):
        super(RLDataset, self).__init__()
        self.data = []
        # 因为只需要问题，所以有好多重复的问题，根据prompt过滤一下
        # hash去重
        seen_hashes = set()
        with jsonlines.open(data_path) as reader:
            for obj in reader:
                prompt = obj["prompt"]
                # 生成查询的哈希值
                prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
                # 如果哈希值出现过，跳过这条数据
                if prompt_hash in seen_hashes:
                    continue
                self.data.append(obj)
                seen_hashes.add(prompt_hash)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system = system
        
        for ex in self.data[:5]:
            self.preprocessing(ex, debug=True)
        
        print("RL Data loading is completed. data len: ", len(self.data))

    def preprocessing(self, example, debug=False):
        """
        示例：
            [gMASK]sop <|system|>
            你是由wdndev开发的个人助手。
            <|user|>
            为以下地点提供纬度和经度：- 纽约市
            - 巴黎
            - 北京
            <|assistant|>
            - 纽约市：纬度：40.7128°N，经度：74.0060°W
            - 巴黎：纬度：48.8566°N，经度：2.3522°E
            - 北京：纬度：39.9042°N，经度：116.4074°E
        """
        query, input_ids = [], []
        prompt_txt = self.system
        # print(type(example))
        user_txt = example["prompt"]

        query = "[gMASK]sop <|system|>\n"  + prompt_txt + "\n" + "<|user|>\n" + user_txt.strip() + "\n" + "<|assistant|>\n"

        instruction = self.tokenizer.encode(text="\n".join(["<|system|>", prompt_txt.strip(), 
                                    "<|user|>", user_txt.strip(), 
                                    "<|assistant|>"]).strip() + "\n",
                                    add_special_tokens=True, 
                                    truncation=True, 
                                    max_length=self.max_length)

        if debug:
            print("query: ", query)
            token_query = self.tokenizer.decode(instruction)
            print("query: ", token_query)
            print(len(query))
            print(len(token_query))
            print("-------------------------------")
        
        input_ids = instruction
        # pad_len = self.max_length - len(input_ids)
        # input_ids += [self.tokenizer.pad_token_id] * pad_len

        input_ids = torch.LongTensor(input_ids)

        return {
            "query": query,
            "input_ids": input_ids,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # return self.preprocessing(self.data[idx])
        processed_example = self.preprocessing(self.data[idx])
        while processed_example is None:
            idx = (idx + 1) % len(self.data)  # 循环至下一个有效样本
            processed_example = self.preprocessing(self.data[idx])

        return processed_example

def load_ppo_dataset(
    data_path: str,
    tokenizer,
    max_length=256,
    sanity_check: bool = False,
    num_proc=24,
    system: str = "你是由wdndev开发的个人助手。",
) -> datasets.Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    dataset = load_dataset(
        'json',
        data_files=data_path,
        split="train",
    )
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def preprocess_function(examples) -> Dict[str, str]:
        # assistant_chosen_txt = examples["chosen"]
        # assistant_rejected_txt = examples["rejected"]

        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["prompt"]:
            query ="\n".join(["<|system|>", system.strip(), 
                            "<|user|>", question.strip(), 
                            "<|assistant|>"]).strip() + "\n"
            
            tokenized_query = tokenizer(query, truncation=True)
            # print("wwwwwwwww: ", tokenized_query)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_query["input_ids"])

        return new_examples
        
        # return {
        #     "prompt": ["Question: " + question + "\n\nAnswer: " for question in examples["prompt"]],
        #     "chosen": examples["chosen"],
        #     "rejected": examples["rejected"],
        # }

    dataset_map = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

    dataset_map = dataset_map.filter(
        lambda x: len(x["input_ids"]) <= max_length
    )

    dataset_map.set_format(type="torch")

    return dataset_map

def load_dpo_dataset(
    data_path: str,
    max_length=256,
    sanity_check: bool = False,
    num_proc=24,
    system: str = "你是由wdndev开发的个人助手。",
) -> datasets.Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    dataset = load_dataset(
        'json',
        data_files=data_path,
        split="train",
        # cache_dir=cache_dir,
        # data_dir=data_dir,
    )
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def preprocess_function(examples) -> Dict[str, str]:
        prompt_txt = system
        # assistant_chosen_txt = examples["chosen"]
        # assistant_rejected_txt = examples["rejected"]

        prompt_list = []
        for question in examples["prompt"]:
            prompt ="\n".join(["<|system|>", system.strip(), 
                            "<|user|>", question.strip(), 
                            "<|assistant|>"]).strip() + "\n"
            prompt_list.append(prompt)

        return {
            "prompt": prompt_list,
            "chosen": examples["chosen"],
            "rejected": examples["rejected"],
        }
        
        # return {
        #     "prompt": ["Question: " + question + "\n\nAnswer: " for question in examples["prompt"]],
        #     "chosen": examples["chosen"],
        #     "rejected": examples["rejected"],
        # }

    dataset_map = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

    dataset_map = dataset_map.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= max_length
        and len(x["prompt"]) + len(x["rejected"]) <= max_length
    )

    # dataset_map.set_format(type="torch")

    return dataset_map

if __name__=="__main__1":
    tokenizer = ChatGLMTokenizer(vocab_file='utils/chatglm3_tokenizer/tokenizer.model')

    # sft_data_path = "data/sft_train/sft_data_test.jsonl"
    # sft_ds = SFTDataset(sft_data_path, tokenizer, max_length=512)

    # rm_data_path = "data/rm_train/rm_data.jsonl"
    # rm_ds = RMDataset(rm_data_path, tokenizer, max_length=512)

    rl_data_path = "data/rm_train/rm_data.jsonl"
    # rl_ds = DPODataset(rl_data_path, tokenizer, max_length=512)

    rl_ds = load_ppo_dataset(rl_data_path, tokenizer, max_length=512, sanity_check=True)
    # rl_ds = load_dpo_dataset(rl_data_path, max_length=512, sanity_check=True)

    train_loader = torch.utils.data.DataLoader(
        rl_ds,
        batch_size=2,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=8,
    )
    for i, item in enumerate(train_loader):
        print(item)
        # print(type(item["input_ids"]))
        # print(item["input_ids"])
        # print(item["labels"])
        # print(item["attention_mask"])
        # print(Y[0])
        # print(loss_mask[0])
        break

if __name__=="__main__":
    tokenizer = ChatGLMTokenizer(vocab_file='utils/chatglm3_tokenizer/tokenizer.model')

    ptm_data_dir = "data/pre_train"
    def get_bin_files_abs_paths(directory):
        bin_files_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.bin'):
                    bin_files_paths.append(os.path.abspath(os.path.join(root, file)))
        return bin_files_paths
    # data_path_list = glob.glob(os.path.join(script_args.dataset_dir_or_path, '*.bin'))
    data_path_list = get_bin_files_abs_paths(ptm_data_dir)
    if len(data_path_list) == 0:
        logger.error("***************NO INPUT DATA********************")
    
    train_ds = PTMDatasetMap(data_path_list, max_length = 512)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=32,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=8,
    )
    print(len(train_loader))
    # for i, item in enumerate(train_loader):
    #     print(item)
    #     if i > 10:
    #         break
    