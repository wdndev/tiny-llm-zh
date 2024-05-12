import json
import os
import glob
import numpy as np
from tqdm import tqdm
import pandas as pd
import csv


def merge_datsets(input_dir):
    total_lines = []
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
                    
                    data_dict = {
                        "prompt": prompt_text,
                        "chosen": chosen_text,
                        "rejected": rejected_text
                    }
                    
                    processed_line = json.dumps(data_dict, ensure_ascii=False) + '\n' 
                    total_lines.append(processed_line)

            if file.endswith('.parquet'):
                # 获取当前文件的绝对路径
                file_path = os.path.join(subdir, file)
                print(file_path)
                # 读取jsonl文件
                df = pd.read_parquet(file_path)
                
                for idx, row in tqdm(df.iterrows(), total=len(df)):
                    prompt_text = row['prompt']
                    chosen_text = row['chosen']
                    rejected_text = row['rejected']
                    
                    data_dict = {
                        "prompt": prompt_text,
                        "chosen": chosen_text,
                        "rejected": rejected_text
                    }
                    
                    processed_line = json.dumps(data_dict, ensure_ascii=False) + '\n' 
                    total_lines.append(processed_line)
    
            if file.endswith('.tsv'):
                # 获取当前文件的绝对路径
                file_path = os.path.join(subdir, file)
                print(file_path)
                # 读取jsonl文件
                df = pd.read_csv(file_path, sep='\t')
                
                for idx, row in tqdm(df.iterrows(), total=len(df)):
                    prompt_text = row['prompt']
                    chosen_text = row['chosen']
                    rejected_text = row['rejected']

                    data_dict = {
                        "prompt": prompt_text,
                        "chosen": chosen_text,
                        "rejected": rejected_text
                    }
                    
                    processed_line = json.dumps(data_dict, ensure_ascii=False) + '\n' 
                    total_lines.append(processed_line)
                    
    # 如果输出子文件夹不存在，则创建它
    output_subfolder = "data/rl_train"
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    # 保存处理后的csv文件到对应的输出子文件夹
    output_file_path = os.path.join(output_subfolder, "rl_data.jsonl")
    # 将处理后的json对象写入新的jsonl文件
    with open(output_file_path, 'w') as outfile:
        for line in total_lines:
            outfile.write(line)


if __name__=="__main__":
    merge_datsets("corpus/rm_train")

