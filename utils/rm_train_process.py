import json
import os
import glob
import numpy as np
from tqdm import tqdm
from chatglm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
import pandas as pd
import csv
import pickle

def process_cvalue_comparsion(input_dir, tokenizer):
    """ https://www.modelscope.cn/datasets/iic/CValues-Comparison/summary
    """
    token_ids = []
    i = 0
    for subdir, dirs, files in os.walk(input_dir):
        for idx, file in enumerate(files):
            # 只处理txt文件
            if file.endswith('.jsonl'):
                # 获取当前文件的绝对路径
                file_path = os.path.join(subdir, file)
                print(file_path)
                # 读取jsonl文件
                with open(file_path, 'r', encoding='utf-8') as infile:
                    lines = infile.readlines()
                    
                for line in tqdm(lines):
                    json_obj = json.loads(line)  # 解析json字符串为python对象
                    
                    prompt_text = json_obj["prompt"]
                    chosen_text = json_obj["pos_resp"]
                    rejected_text = json_obj["neg_resp"]
                    
                    prompt_id = tokenizer.encode(prompt_text, add_special_tokens=False)
                    chosen_id = tokenizer.encode(chosen_text, add_special_tokens=False)
                    rejected_id = tokenizer.encode(rejected_text, add_special_tokens=False)
                    
                    text_ids_j = prompt_id + [tokenizer.special_tokens['<bos>']] + chosen_id + [tokenizer.special_tokens['<eos>']]
                    text_ids_k = prompt_id + [tokenizer.special_tokens['<bos>']] + rejected_id + [tokenizer.special_tokens['<eos>']]

                    if len(text_ids_j) > 5 and len(text_ids_k) > 5:
                        token_ids.append([text_ids_j, text_ids_k])

    return token_ids

def process_reward_single(input_dir, tokenizer):
    """ https://huggingface.co/datasets/beyond/rlhf-reward-single-round-trans_chinese
    """

    token_ids = []
    i = 0
    # 使用glob找出文件夹下所有的.parquet
    for file in glob.glob(os.path.join(input_dir, '*.parquet')):
        # 读取jsonl文件
        print(file)
        # 读取parquet文件
        df = pd.read_parquet(file)
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            prompt_text = row['prompt']
            chosen_text = row['chosen']
            rejected_text = row['rejected']

            prompt_id = tokenizer.encode(prompt_text, add_special_tokens=False)
            chosen_id = tokenizer.encode(chosen_text, add_special_tokens=False)
            rejected_id = tokenizer.encode(rejected_text, add_special_tokens=False)
            
            text_ids_j = prompt_id + [tokenizer.special_tokens['<bos>']] + chosen_id + [tokenizer.special_tokens['<eos>']]
            text_ids_k = prompt_id + [tokenizer.special_tokens['<bos>']] + rejected_id + [tokenizer.special_tokens['<eos>']]

            if len(text_ids_j) > 5 and len(text_ids_k) > 5:
                token_ids.append([text_ids_j, text_ids_k])
    
    return token_ids


def process_zhihu_rlhf(input_dir, tokenizer):
    """ https://huggingface.co/datasets/liyucheng/zhihu_rlhf_3k
    """
    token_ids = []
    i = 0
    # 使用glob找出文件夹下所有的.tsv
    for file in glob.glob(os.path.join(input_dir, '*.tsv')):
        # 读取jsonl文件
        print(file)
        # 读取 tsv 文件
        df = pd.read_csv(file, sep='\t')
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            prompt_text = row['prompt']
            chosen_text = row['chosen']
            rejected_text = row['rejected']

            prompt_id = tokenizer.encode(prompt_text, add_special_tokens=False)
            chosen_id = tokenizer.encode(chosen_text, add_special_tokens=False)
            rejected_id = tokenizer.encode(rejected_text, add_special_tokens=False)
            
            text_ids_j = prompt_id + [tokenizer.special_tokens['<bos>']] + chosen_id + [tokenizer.special_tokens['<eos>']]
            text_ids_k = prompt_id + [tokenizer.special_tokens['<bos>']] + rejected_id + [tokenizer.special_tokens['<eos>']]

            if len(text_ids_j) > 5 and len(text_ids_k) > 5:
                token_ids.append([text_ids_j, text_ids_k])
    
    return token_ids


if __name__=="__main__":
    tokenizer = ChatGLMTokenizer(vocab_file='utils/chatglm3_tokenizer/tokenizer.model')
    
    token_ids = process_cvalue_comparsion("corpus/rm_train/cvalue_comparison", tokenizer)
    print("cvalue_comparison: ", len(token_ids))
    reward_single_token_ids = process_reward_single("corpus/rm_train/reward_single", tokenizer)
    print("reward_single: ", len(reward_single_token_ids))
    
    token_ids.extend(reward_single_token_ids)
    
    zhihu_token_ids = process_zhihu_rlhf("corpus/rm_train/zhihu", tokenizer)
    print("zhihu: ", len(zhihu_token_ids))
    
    token_ids.extend(zhihu_token_ids)

    print("all: ", len(token_ids))
    
    # arr = np.array(list(token_ids), dtype = object)
    output_file_path = "data/rm_train/rm_data.bin"
    with open(output_file_path, 'wb') as f:
        # f.write(arr.tobytes())
        pickle.dump(token_ids, f)

if __name__=="__main__1":
    # df=df.sample(frac=1.0)
    # output_file_path = "data/rm_train/rm_data.bin"
    # with open(output_file_path,'rb') as f:
    #     token_ids = pickle.load(f)
    # print("all: ", len(token_ids))
    # print(type(token_ids))
    # print(type(token_ids[0]))
    # print(len(token_ids[0][0]))
    # print(len(token_ids[1][1]))
    # print(token_ids[1])
    tokenizer = ChatGLMTokenizer(vocab_file='utils/chatglm3_tokenizer/tokenizer.model')
    
    print(tokenizer.special_tokens['<bos>'])
    print(tokenizer.special_tokens['<eos>'])
    
    # input_id = token_ids[1][1]
    # context_length = input_id.index(tokenizer.special_tokens['<bos>'])
    # print(context_length)