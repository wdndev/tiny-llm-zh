import pickle
import jsonlines
import hashlib
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from sklearn.model_selection import train_test_split
from utils.chatglm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer

class PretrainDataset(Dataset):
    def __init__(self, data_path_list, max_length=256, memmap=False):
        super().__init__()
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

        print("memmap:{} train data.shape:{}".format(memmap, self.data.shape))
        print("downloading finished.....")
        
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index: int):
        #
        sample = self.data[index]
        X=np.array(sample[:-1]).astype(np.int64)
        Y=np.array(sample[1:]).astype(np.int64)
        
        return torch.from_numpy(X), torch.from_numpy(Y)
    

class SFTDataset(Dataset):
    def __init__(self, data_path_list, max_length=256, tokenizer=None):
        self.data = []
        for data_path in data_path_list:
            with open(data_path,'rb') as f:
                data = pickle.load(f)
                self.data.extend(data)
                
        print("ori len: ", len(self.data))
        # 原地删除长度超过max_length的子列表
        i = 0
        while i < len(self.data):
            if len(self.data[i]) > max_length:
                del self.data[i]
            else:
                i += 1
        
        print("process len: ", len(self.data))
        
        # self.data = np.concatenate(data_list)
        
        self.max_length = max_length
        
        if tokenizer is None:
            self.tokenizer = ChatGLMTokenizer(vocab_file='utils/chatglm3_tokenizer/tokenizer.model')
        else:
            self.tokenizer = tokenizer
        self.bos=self.tokenizer.special_tokens['<bos>']
        self.eos=self.tokenizer.special_tokens['<eos>']
        self.pad=0  #self.tokenizer.special_tokens['<pad>']
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index : int):
        input_id = self.data[index]
        
        context_length = input_id.index(self.bos)
        mask_position = context_length - 1
        pad_len = self.max_length - len(input_id)
        input_id = input_id + [self.pad] * pad_len
        
        # 生成一个mask向量（loss_mask），这个向量将作为损失函数计算过程中的权重向量，
        # 决定哪些部分的预测值得到计算损失，哪些部分忽略损失计算
        if pad_len == 0:
            # 当pad_len == 0时，说明没有填充或者填充部分不需要考虑在内
            # 1.先填充context_length个0，代表mask位置之前的有效词汇不参与损失计算。
            # 2.接着填充从mask位置下一个位置开始到序列结束的所有位置为1，这些位置是需要计算损失的。
            # 3.最后补足pad_len个0，但实际上这部分由于pad_len为0所以不起作用。
            loss_mask = [0]*context_length + [1]*(len(input_id[mask_position+1:])) + [0]*pad_len
        else:
            # 当pad_len != 0时，表示序列尾部有填充部分需要排除在损失计算之外：
            # 1.先填充context_length个0，代表mask位置之前的有效词汇不参与损失计算。
            # 2.接着填充从mask位置下一个位置开始到序列尾部减去pad_len的所有位置为1，这些位置是需要计算损失的。
            # 3.最后补足pad_len个0，这部分对应序列尾部的填充部分，它们同样不参与损失计算。
            loss_mask = [0]*context_length + [1]*(len(input_id[mask_position+1:-pad_len])) + [0]*pad_len
        
        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[:-1])
        #
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)


class RMDataset(Dataset):
    def __init__(self, data_path_list, max_length=256, tokenizer=None):
        self.data = []
        for data_path in data_path_list:
            with open(data_path,'rb') as f:
                data = pickle.load(f)
                self.data.extend(data)
                
        print("ori len: ", len(self.data))
        # 原地删除长度超过max_length的子列表
        i = 0
        while i < len(self.data):
            if len(self.data[i][0]) > max_length or len(self.data[i][1]) > max_length:
                del self.data[i]
            else:
                i += 1
        
        print("process len: ", len(self.data))
        
        # self.data = np.concatenate(data_list)
        
        self.max_length = max_length
        
        if tokenizer is None:
            self.tokenizer = ChatGLMTokenizer(vocab_file='utils/chatglm3_tokenizer/tokenizer.model')
        else:
            self.tokenizer = tokenizer
        self.bos=self.tokenizer.special_tokens['<bos>']
        self.eos=self.tokenizer.special_tokens['<eos>']
        self.pad=0  #self.tokenizer.special_tokens['<pad>']
        
    def __len__(self):
        return len(self.data)
    
    def process_ids(self, input_id):
        context_length = input_id.index(self.bos)
        mask_position = context_length - 1
        pad_len = self.max_length - len(input_id)
        input_id = input_id + [self.pad] * pad_len
        
        # 生成一个mask向量（loss_mask），这个向量将作为损失函数计算过程中的权重向量，
        # 决定哪些部分的预测值得到计算损失，哪些部分忽略损失计算
        if pad_len == 0:
            # 当pad_len == 0时，说明没有填充或者填充部分不需要考虑在内
            # 1.先填充context_length个0，代表mask位置之前的有效词汇不参与损失计算。
            # 2.接着填充从mask位置下一个位置开始到序列结束的所有位置为1，这些位置是需要计算损失的。
            # 3.最后补足pad_len个0，但实际上这部分由于pad_len为0所以不起作用。
            loss_mask = [0]*context_length + [1]*(len(input_id[mask_position+1:])) + [0]*pad_len
        else:
            # 当pad_len != 0时，表示序列尾部有填充部分需要排除在损失计算之外：
            # 1.先填充context_length个0，代表mask位置之前的有效词汇不参与损失计算。
            # 2.接着填充从mask位置下一个位置开始到序列尾部减去pad_len的所有位置为1，这些位置是需要计算损失的。
            # 3.最后补足pad_len个0，这部分对应序列尾部的填充部分，它们同样不参与损失计算。
            loss_mask = [0]*context_length + [1]*(len(input_id[mask_position+1:-pad_len])) + [0]*pad_len
        
        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[:-1])
        #
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)
    
    def __getitem__(self, index : int):
        input_id = self.data[index]
        
        X_j, Y_j, loss_mask_j = self.process_ids(input_id[0])
        X_k, Y_k, loss_mask_k = self.process_ids(input_id[1])
        #
        return X_j, Y_j, loss_mask_j, X_k, Y_k, loss_mask_k

class RLDataset(Dataset):
    def __init__(self, data_path, max_length=256, tokenizer=None):
        self.max_length = max_length
        if tokenizer is None:
            self.tokenizer = ChatGLMTokenizer(vocab_file='utils/chatglm3_tokenizer/tokenizer.model')
        else:
            self.tokenizer = tokenizer
        self.data = []
        
        # # 加载 JSONL 数据
        # json_data = []
        # with jsonlines.open(data_path) as reader:
        #     for obj in reader:
        #         json_data.append(obj)
        # # 加载 JSONL 数据并去重
        # seen_hashes = set()
        # for example in json_data:
        #     query = example["prompt"]
        #     query_hash = hashlib.md5(query.encode()).hexdigest()  # 生成查询的哈希值
        #     query_id = self.tokenizer.encode(query, add_special_tokens=False)
        #     if len(query_id) > self.max_length:
        #         continue
            
        #     data_example = {
        #         "query": query,
        #         "input_ids": query_id,
        #     }
            
        #     self.data.append(data_example)
        
        # 加载 JSONL 数据并去重
        seen_hashes = set()
        with jsonlines.open(data_path) as reader:
            for obj in reader:
                query = obj["prompt"]
                query_hash = hashlib.md5(query.encode()).hexdigest()  # 生成查询的哈希值
                if query_hash in seen_hashes:  # 如果哈希值未出现过
                    continue
                query_id = self.tokenizer.encode(query, add_special_tokens=False)
                if len(query_id) > self.max_length:
                    continue
                data_example = {
                    "query": query,
                    "input_ids": query_id,
                }
                self.data.append(data_example)
                seen_hashes.add(query_hash)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "query": item["query"],
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
        }

#
if __name__=="__main__":
    rl_data_path = "data/rl_train/rl_data.jsonl"
    tokenizer = ChatGLMTokenizer(vocab_file='utils/chatglm3_tokenizer/tokenizer.model')
    
    rl_train_ids = RLDataset(rl_data_path, max_length=512)
    
    print(len(rl_train_ids))  # 29133 ,9693, 173986, 75497
    # print(sft_train_ids[10])
    
    train_loader = torch.utils.data.DataLoader(
        rl_train_ids,
        batch_size=1,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=8,
    )
    print("333333333")
    for i, item in enumerate(train_loader):
        print(type(item["query"]))
        print(item["query"])
        print(item["input_ids"])
        print(type(item["query"][0]))
        # print(Y[0])
        # print(loss_mask[0])
        break
    
    