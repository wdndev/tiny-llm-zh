import json
import os
import glob
import numpy as np
from tqdm import tqdm
from chatglm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
import pandas as pd
import csv
import pickle

#from zhconv import convert

def process_bell_2m(file_path, tokenizer):
    """ https://huggingface.co/datasets/BelleGroup/train_2M_CN
    """

    token_ids = []
    with open(file_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    for line in tqdm(lines):
        json_obj = json.loads(line)  # 解析json字符串为python对象
        
        instruction = json_obj["instruction"]
        input_str = json_obj["input"]
        answer = json_obj["output"]
        
        question = instruction + input_str
        
        if len(question) < 10 and len(answer) < 1:
            continue
        
        prompt_id = tokenizer.encode(question, add_special_tokens=False)
        answer_id = tokenizer.encode(answer, add_special_tokens=False)
        
        text_id = prompt_id + [tokenizer.special_tokens['<bos>']] + answer_id + [tokenizer.special_tokens['<eos>']]

        if len(text_id) > 5:
            token_ids.append(text_id)
        
    return token_ids

def process_nlp(file_path, tokenizer):
    """ https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M
    """
    token_ids = []
    with open(file_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    for line in tqdm(lines):
        json_obj = json.loads(line)  # 解析json字符串为python对象
        
        # instruction = json_obj["instruction"]
        question = json_obj["input"]
        answer = json_obj["target"]
        
        if len(question) < 10 and len(answer) < 1:
            continue
        
        prompt_id = tokenizer.encode(question, add_special_tokens=False)
        answer_id = tokenizer.encode(answer, add_special_tokens=False)
        
        text_id = prompt_id + [tokenizer.special_tokens['<bos>']] + answer_id + [tokenizer.special_tokens['<eos>']]

        if len(text_id) > 5:
            token_ids.append(text_id)
        
    return token_ids

def process_tigerbot_sft(input_dir, tokenizer):
    token_ids = []
    for subdir, dirs, files in os.walk(input_dir):
        for idx, file in enumerate(files):
            # 只处理txt文件
            if file.endswith('.json'):
                # 获取当前文件的绝对路径
                file_path = os.path.join(subdir, file)
                print(file_path)
                # 读取jsonl文件
                with open(file_path, 'r', encoding='utf-8') as infile:
                    lines = infile.readlines()
                    
                for line in tqdm(lines):
                    json_obj = json.loads(line)  # 解析json字符串为python对象
                    
                    instruction = json_obj["instruction"]
                    input_str = json_obj["input"]
                    answer = json_obj["output"]
                    
                    question = instruction + input_str
                    
                    if len(question) < 10 and len(answer) < 1:
                        continue
                    
                    prompt_id = tokenizer.encode(question, add_special_tokens=False)
                    answer_id = tokenizer.encode(answer, add_special_tokens=False)
                    
                    text_id = prompt_id + [tokenizer.special_tokens['<bos>']] + answer_id + [tokenizer.special_tokens['<eos>']]

                    if len(text_id) > 5:
                        token_ids.append(text_id)

    return token_ids


if __name__=="__main__":
    tokenizer = ChatGLMTokenizer(vocab_file='chatglm3_tokenizer/tokenizer.model')
    
    token_ids = process_bell_2m("sft_train/bell_2m/train_2M_CN.json", tokenizer)
    print("bell 2m: ", len(token_ids))
    nlp_token_ids = process_nlp("sft_train/nlp/firefly-train-1.1M.jsonl", tokenizer)
    print("nlp: ", len(nlp_token_ids))
    
    token_ids.extend(nlp_token_ids)
    
    tigerbot_token_ids = process_tigerbot_sft("sft_train/tigerbot", tokenizer)
    print("tigerbot: ", len(tigerbot_token_ids))
    
    token_ids.extend(tigerbot_token_ids)

    print("all: ", len(token_ids))
    
    # arr = np.array(list(token_ids), dtype = object)
    output_file_path = "sft_train/sft_data.bin"
    with open(output_file_path, 'wb') as f:
        # f.write(arr.tobytes())
        pickle.dump(token_ids, f)

if __name__=="__main__1":
    # df=df.sample(frac=1.0)
    output_file_path = "sft_train/sft_data.bin"
    with open(output_file_path,'rb') as f:
        token_ids = pickle.load(f)
    print("all: ", len(token_ids))
    print(type(token_ids))
    print(type(token_ids[0]))
    print(len(token_ids[0]))
    print(len(token_ids[1]))
    # print(token_ids[1])
    tokenizer = ChatGLMTokenizer(vocab_file='utils/chatglm3_tokenizer/tokenizer.model')
    
    input_id = token_ids[1]
    context_length = input_id.index(tokenizer.special_tokens['<bos>'])
    print(context_length)