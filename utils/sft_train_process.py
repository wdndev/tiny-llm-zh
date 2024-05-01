import json
import os
import glob
import numpy as np
from tqdm import tqdm
import pandas as pd
import csv
import pickle
import json

#from zhconv import convert

def process_bell_2m(file_path):
    """ https://huggingface.co/datasets/BelleGroup/train_2M_CN
    """

    total_lines = []
    with open(file_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    for line in tqdm(lines):
        json_obj = json.loads(line)  # 解析json字符串为python对象
        
        instruction = json_obj["instruction"]
        input_str = json_obj["input"]
        answer = json_obj["output"]
        
        question = instruction + input_str

        data_dict = {
            "question": question,
            "answer": answer
        }

        processed_line = json.dumps(data_dict, ensure_ascii=False) + '\n' 
        total_lines.append(processed_line)
        
    return total_lines

def process_nlp(file_path):
    """ https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M
    """
    total_lines = []
    with open(file_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    for line in tqdm(lines):
        json_obj = json.loads(line)  # 解析json字符串为python对象
        
        # instruction = json_obj["instruction"]
        question = json_obj["input"]
        answer = json_obj["target"]

        data_dict = {
            "question": question,
            "answer": answer
        }

        processed_line = json.dumps(data_dict, ensure_ascii=False) + '\n' 
        total_lines.append(processed_line)
        
    return total_lines

def process_tigerbot_sft(input_dir):
    """ https://huggingface.co/datasets/TigerResearch/sft_zh
    """
    total_lines = []
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

                    data_dict = {
                        "question": question,
                        "answer": answer
                    }

                    processed_line = json.dumps(data_dict, ensure_ascii=False) + '\n' 
                    total_lines.append(processed_line)
    
    return total_lines


if __name__=="__main__":

    total_lines = process_bell_2m("corpus/sft_train/bell_2m/train_2M_CN.json")
    print("bell 2m: ", len(total_lines))
    nlp_total_lines = process_nlp("corpus/sft_train/nlp/firefly-train-1.1M.jsonl")
    print("nlp: ", len(nlp_total_lines))
    
    total_lines.extend(nlp_total_lines)
    
    tigerbot_total_lines = process_tigerbot_sft("corpus/sft_train/tigerbot")
    print("tigerbot: ", len(tigerbot_total_lines))
    
    total_lines.extend(tigerbot_total_lines)

    print("all: ", len(total_lines))

    # 如果输出子文件夹不存在，则创建它
    output_subfolder = "data/sft_train"
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    # 保存处理后的csv文件到对应的输出子文件夹
    output_file_path = os.path.join(output_subfolder, "sft_data_test.jsonl")
    # 将处理后的json对象写入新的jsonl文件
    with open(output_file_path, 'w') as outfile:
        for line in total_lines:
            outfile.write(line)


