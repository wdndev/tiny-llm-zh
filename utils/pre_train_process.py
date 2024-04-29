import json
import os
import glob
import numpy as np
from tqdm import tqdm
from chatglm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
import pandas as pd
#from zhconv import convert
def process_wiki_clean(file_path, tokenizer):
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
        responses = df['content']
        
        for text in tqdm(responses):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(tokenizer.special_tokens['<eos>'])
            if len(tokens) > 5:
                all_tokens += tokens
    arr = np.array(all_tokens, dtype=np.uint16)
    # base_name, ext = os.path.splitext(file_path)
    output_file_path = "tigerbot_part" + '.bin'
    with open(output_file_path, 'wb') as f:
        f.write(arr.tobytes())                
                    
def process_zhihu(input_dir, tokenizer):
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
            

def process_medical(data_path,name):
    f=open(data_path,'r',encoding='utf-8')
    doc_ids=[]
    while True:
        line=f.readline()
        if not line:
            break
        line=json.loads(line)
        text=line['text']
        text_id=tokenizer.encode(text,add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id)>5:
            doc_ids+=text_id
    arr = np.array(doc_ids,dtype=np.uint16)
    with open('./data/medical_{}.bin'.format(name),'wb') as f:
        f.write(arr.tobytes()) 

def sft_to_pretrain():
    doc_ids=[]

    '''
    df=pd.read_csv('./data/medical_qa_144w.csv')
    for _,q,a in tqdm(df.itertuples()):
        q_id = tokenizer.encode(q,add_special_tokens=False)
        a_id = tokenizer.encode(a,add_special_tokens=False)
        #
        print(q)
        print(a)
        print('-----')
        text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
        if len(text_id)>5:
            doc_ids+=text_id
    '''

    with open('./data/shibing624_medical/finetune/train_en_1.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id
    with open('./data/shibing624_medical/finetune/test_en_1.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id
    with open('./data/shibing624_medical/finetune/valid_en_1.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id

    with open('./data/shibing624_medical/finetune/train_zh_0.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['instruction']+line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id
    with open('./data/shibing624_medical/finetune/test_zh_0.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['instruction']+line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id
    with open('./data/shibing624_medical/finetune/valid_zh_0.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['instruction']+line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id

    arr = np.array(doc_ids,dtype=np.uint16)
    print(arr.shape)
    with open('./data/medical_qa.bin','wb') as f:
        f.write(arr.tobytes())

def process_baidu(tokenizer):
    BATCH_SIZE = 1000000

    cnt = 0
    batch_cnt = 0
    token = 0
    doc_ids = []

    f1 = open('baidubaike/563w_baidubaike.json', 'r', encoding='utf-8')
    
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
    
def process_c4():
    c4_zh_paths = glob.glob('./data/c4_zh/*')
    c4_zh_paths=sorted(c4_zh_paths)
    print(len(c4_zh_paths))
    cnt=0
    token=0
    doc_ids=[]
    for per in tqdm(c4_zh_paths):
        with open(per,'r') as f:
            for line in f:
                text = json.loads(line)
                text = text['text']
                text_id=tokenizer.encode(text,add_special_tokens=False)
                text_id.append(tokenizer.special_tokens['<eos>'])
                if len(text_id)>5:
                    doc_ids+=text_id
                cnt+=1

    arr = np.array(doc_ids,dtype=np.uint16)
    with open('./data/c4_zh.bin','wb') as f:
        f.write(arr.tobytes())
    print(arr.shape)

def process_wudao():
    wudao_zh_paths = glob.glob('./data/WuDaoCorpus2.0_base_200G/*')
    wudao_zh_paths=sorted(wudao_zh_paths)
    print(len(wudao_zh_paths))#很多子文件
    cnt=0
    token=0
    doc_ids=[]
    for per in tqdm(wudao_zh_paths[320:]):#wudao_zh_paths[i:j]手动分片，一片片处理，不然太大一次性处理不完
        with open(per,'r') as f:
            data=json.load(f)
            for text in data:
                text = text['title'] + text['content']
                text_id=tokenizer.encode(text,add_special_tokens=False)
                text_id.append(tokenizer.special_tokens['<eos>'])
                if len(text_id)>5:
                    doc_ids+=text_id
                #
                # if cnt%10000==0:
                #     print(cnt)
                cnt+=1
                #token+=len(text_id)
                #break
        #
        # arr = np.array(doc_ids,dtype=np.uint16)
        # with open('./data/c4-zh/{}.bin'.format(per.split('/')[-1].split('.')[0]),'wb') as f:
        #     f.write(arr.tobytes())
        # print(arr.shape)
    arr = np.array(doc_ids,dtype=np.uint16)
    with open('./data/wudaocorpus_zh_16.bin','wb') as f:
        f.write(arr.tobytes())
    print(arr.shape)

def merge_bin(data_path_list : list):
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
    tokenizer = ChatGLMTokenizer(vocab_file='utils/chatglm3_tokenizer/tokenizer.model')
    
    # process_webnovel("webnovel-chinese/data", tokenizer)
    # process_wiki_clean("wiki_cn/wikipedia-cn.json", tokenizer)
    # process_zhihu("./zhihu", tokenizer)
    process_tigerbot_part("tigerbot", tokenizer)
    # process_baidu(tokenizer)
    # 数据预处理-如果下载分词处理后的数据，可以不用执行以下函数
    # process_wiki_clean()
    # process_medical('./data/shibing624_medical/pretrain/medical_book_zh.json','book')
    # process_medical('./data/shibing624_medical/pretrain/train_encyclopedia.json','encyclopedia')
    # process_baidu()
    # process_c4()
    # process_wudao()

    # print('data processing finished!')

    # # 分词处理后的文件列表
    # data_path_list=[
    #     './data/baidubaike_563w_1.bin',
    #     './data/baidubaike_563w_2.bin',
    #     './data/baidubaike_563w_3.bin',
    #     './data/baidubaike_563w_4.bin',
    #     './data/baidubaike_563w_5.bin',
    #     './data/medical_book.bin',
    #     './data/medical_encyclopedia.bin',
    #     './data/wiki.bin',
    #     './data/c4_zh_0.bin',
    #     './data/c4_zh_1.bin',
    #     './data/c4_zh_2.bin',
    #     './data/c4_zh_3.bin',
    #     './data/c4_zh_4.bin',
    #     './data/c4_zh_5.bin',
    #     './data/c4_zh_6.bin',
    #     './data/c4_zh_7.bin',
    #     './data/c4_zh_8.bin',
    #     './data/wudaocorpus_zh_0.bin',
    #     './data/wudaocorpus_zh_1.bin',
    #     './data/wudaocorpus_zh_2.bin',
    #     './data/wudaocorpus_zh_3.bin',
    #     './data/wudaocorpus_zh_4.bin',
    #     './data/wudaocorpus_zh_5.bin',
    #     './data/wudaocorpus_zh_6.bin',
    #     './data/wudaocorpus_zh_7.bin',
    #     './data/wudaocorpus_zh_8.bin',
    #     './data/wudaocorpus_zh_9.bin',
    #     './data/wudaocorpus_zh_10.bin',
    #     './data/wudaocorpus_zh_11.bin',
    #     './data/wudaocorpus_zh_12.bin',
    #     './data/wudaocorpus_zh_13.bin',
    #     './data/wudaocorpus_zh_14.bin',
    #     './data/wudaocorpus_zh_15.bin',
    #     './data/wudaocorpus_zh_16.bin',
    # ]
    # data_lst=[]
    # for data_path in tqdm(data_path_list):
    #     with open(data_path,'rb') as f:
    #         data=np.fromfile(f,dtype=np.uint16)
    #         data_lst.append(data)
    # arr = np.concatenate(data_lst)
    # print(arr.shape)
    # with open('./data/pretrain_data.bin','wb') as f:
    #     f.write(arr.tobytes())