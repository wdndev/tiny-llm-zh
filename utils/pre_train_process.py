import json
import os
import glob
import numpy as np
from tqdm import tqdm
from utils.chatglm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
import pandas as pd


def process_wiki_clean(file_path, tokenizer):
    """ https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_tokens = []
    for line in tqdm(data):
        text = line['completion']
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(tokenizer.special_tokens['<eos>'])
        if len(tokens) > 5:
            all_tokens += tokens
    arr = np.array(all_tokens, dtype=np.uint16)
    base_name, ext = os.path.splitext(file_path)
    output_file_path = base_name + '.bin'
    with open(output_file_path, 'wb') as f:
        f.write(arr.tobytes())
        
def process_webnovel(input_dir, tokenizer):
    for subdir, dirs, files in os.walk(input_dir):
        for idx, file in enumerate(files):
            # 只处理txt文件
            if file.endswith('.jsonl'):
                # 获取当前文件的绝对路径
                file_path = os.path.join(subdir, file)
                all_tokens = []
                # 读取jsonl文件
                with open(file_path, 'r', encoding='utf-8') as infile:
                    lines = infile.readlines()
                    
                for line in tqdm(lines):
                    json_obj = json.loads(line)  # 解析json字符串为python对象
                    text = json_obj['text']
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    tokens.append(tokenizer.special_tokens['<eos>'])
                    if len(tokens) > 5:
                        all_tokens += tokens
    
                arr = np.array(all_tokens, dtype = np.uint16)
                base_name, ext = os.path.splitext(file_path)
                output_file_path = base_name + '.bin'
                with open(output_file_path, 'wb') as f:
                    f.write(arr.tobytes())
                    
def process_tigerbot_wiki(input_dir, tokenizer):
    """ https://huggingface.co/datasets/TigerResearch/tigerbot-wiki-plugin
    """
    for subdir, dirs, files in os.walk(input_dir):
        for idx, file in enumerate(files):
            # 只处理txt文件
            if file.endswith('.json'):
                # 获取当前文件的绝对路径
                file_path = os.path.join(subdir, file)
                all_tokens = []
                # 读取jsonl文件
                with open(file_path, 'r', encoding='utf-8') as infile:
                    lines = infile.readlines()
                    
                for line in tqdm(lines):
                    json_obj = json.loads(line)  # 解析json字符串为python对象
                    text = json_obj['text']
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    tokens.append(tokenizer.special_tokens['<eos>'])
                    if len(tokens) > 5:
                        all_tokens += tokens
    
                arr = np.array(all_tokens, dtype = np.uint16)
                base_name, ext = os.path.splitext(file_path)
                output_file_path = base_name + '.bin'
                with open(output_file_path, 'wb') as f:
                    f.write(arr.tobytes())
                    
def process_tigerbot_part(input_dir, tokenizer):
    """ https://huggingface.co/datasets/TigerResearch/pretrain_zh
    """
    # df = pd.read_parquet("zhizhu/train-00000-of-00005-a1278ede4e8c5cdb.parquet")
    # responses = df['RESPONSE']
    # print(len(responses))
    # print(responses[4000])
    all_tokens = []
    total_len = 0
    file_idx = 7
    # 使用glob找出文件夹下所有的.parquet
    for file in glob.glob(os.path.join(input_dir, '*.parquet')):
        # 读取jsonl文件
        print(file)
        # 读取parquet文件
        df = pd.read_parquet(file)
        
        # 提取RESPONSE列
        responses = df['content']
        
        for text in tqdm(responses):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(tokenizer.special_tokens['<eos>'])
            if len(tokens) > 5:
                all_tokens += tokens

        total_len += len(df)
        if total_len > 600000:
            arr = np.array(all_tokens, dtype=np.uint16)
            output_file_path = "tigerbot_part_" + str(file_idx) + '.bin'
            with open(output_file_path, 'wb') as f:
                f.write(arr.tobytes())

            all_tokens = []
            total_len = 0
            file_idx += 1
            
    if len(all_tokens) > 0:
        arr = np.array(all_tokens, dtype=np.uint16)
        output_file_path = "tigerbot_part_" + str(file_idx) + '.bin'
        with open(output_file_path, 'wb') as f:
            f.write(arr.tobytes())
                    
def process_zhihu(input_dir, tokenizer):
    """ https://huggingface.co/datasets/wangrui6/Zhihu-KOL
    """
    # df = pd.read_parquet("zhizhu/train-00000-of-00005-a1278ede4e8c5cdb.parquet")
    # responses = df['RESPONSE']
    # print(len(responses))
    # print(responses[4000])
    all_tokens = []
    # 使用glob找出文件夹下所有的.parquet
    for file in glob.glob(os.path.join(input_dir, '*.parquet')):
        # 读取jsonl文件
        print(file)
        # 读取parquet文件
        df = pd.read_parquet(file)

        # 提取RESPONSE列
        responses = df['RESPONSE']
        
        for text in tqdm(responses):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(tokenizer.special_tokens['<eos>'])
            if len(tokens) > 5:
                all_tokens += tokens
    arr = np.array(all_tokens, dtype=np.uint16)
    # base_name, ext = os.path.splitext(file_path)
    output_file_path = "zhihu" + '.bin'
    with open(output_file_path, 'wb') as f:
        f.write(arr.tobytes())

def process_baidu_baike(input_path, tokenizer):
    """ https://huggingface.co/datasets/xuqinyang/BaiduBaike-5.63M
    """
    BATCH_SIZE = 1000000

    cnt = 0
    batch_cnt = 0
    token = 0
    doc_ids = []

    f1 = open(input_path, 'r', encoding='utf-8')
    
    while True:
        line = f1.readline()
        if not line:
            break
        line = json.loads(line)
        text = ''
        try:
            text += line['title']+'：' + line['summary']
        except:
            pass
        for per in line['sections']:
            text += per['title']+'：'+per['content']+'。'
        text_id = tokenizer.encode(text, add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id) > 5:
            doc_ids += text_id
        cnt += 1
        if cnt % BATCH_SIZE==0:
            batch_cnt += 1
            arr = np.array(doc_ids, dtype=np.uint16)
            doc_ids=[]
            print('cnt:',cnt,'arr_shape:',arr.shape)
            with open('./baidubaike_563w_{}.bin'.format(batch_cnt),'wb') as f2:
                f2.write(arr.tobytes())
            del arr

    if not doc_ids:
        batch_cnt += 1
        arr = np.array(doc_ids, dtype=np.uint16)
        print('cnt:',cnt,'arr_shape:',arr.shape)
        with open('./baidubaike_563w_{}.bin'.format(batch_cnt),'wb') as f:
            f.write(arr.tobytes())

def merge_bin(data_path_list : list):
    """ 合并所有bin文件
    """
    data_arr = []
    for data_path in tqdm(data_path_list):
        with open(data_path,'rb') as f:
            data = np.fromfile(f,dtype = np.uint16)
            data_arr.append(data)
    arr = np.concatenate(data_arr)
    print(arr.shape)
    with open('./data/pretrain_data.bin','wb') as f:
        f.write(arr.tobytes())
    
if __name__=="__main__":
    tokenizer = ChatGLMTokenizer(vocab_file='utils/tokenizer/tokenizer.model')
    
    # process_webnovel("webnovel-chinese/data", tokenizer)
    # process_wiki_clean("corpus/pre_train/wiki_cn/wikipedia-cn.json", tokenizer)
    # process_zhihu("corpus/pre_train/zhihu", tokenizer)
    process_tigerbot_part("corpus/pre_train/tigerbot2", tokenizer)
    # process_baidu_baike('corpus/pre_train/baidubaike/563w_baidubaike.json', tokenizer)