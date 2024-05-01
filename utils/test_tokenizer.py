from utils.chatglm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
import json
import torch

def process_func(prompt_txt:str, user_txt:str, assistant_txt:str, max_length=512):
    input_ids, labels = [], []
    prompt = [tokenizer.get_command("<|system|>")] + tokenizer.encode(prompt_txt + "\n", add_special_tokens=False)
    instruction_ = [tokenizer.get_command("<|user|>")] + tokenizer.encode(user_txt.strip() + "\n", add_special_tokens=False,max_length=max_length) + [tokenizer.get_command("<|assistant|>")]
    instruction = prompt + instruction_
    response = tokenizer.encode(assistant_txt.strip(), add_special_tokens=False)
    input_ids = instruction + response + [tokenizer.eos_token_id]
    labels = [tokenizer.pad_token_id] * len(instruction) + response + [tokenizer.eos_token_id]
    pad_len = max_length - len(input_ids)
    # print()
    input_ids += [tokenizer.pad_token_id] * pad_len
    labels += [tokenizer.pad_token_id] * pad_len
    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

    input_ids = torch.LongTensor(input_ids)
    labels = torch.LongTensor(labels)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }

if __name__=="__main__":
    tokenizer = ChatGLMTokenizer(vocab_file='utils/tokenizer/tokenizer.model')
    
    vocab = tokenizer.get_vocab()
    print(len(vocab))
    print(tokenizer.vocab_size)
    # tokenizer.save_vocabulary(save_directory="test")
    # with open('vocab_utf8.txt', 'w', encoding='utf-8') as f:
    #     json.dump(vocab, f, indent=4)
    text = "家牛的体重范围是多少？" + "\n"
    # encode_1 = tokenizer(text)
    encode_2 = tokenizer.encode(text)

    # print(encode_1)
    print(encode_2)

    # print(tokenizer.decode(encode_1["input_ids"]))
    print(tokenizer.decode(encode_2))
    
    # sys = "你是由wdndev开发的个人助手。"
    # q = "家牛的体重范围是多少？"
    # a = "公牛：800-900千克，母牛：600-700千克"
    
    # inputs = process_func(sys, q, a)
    
    # for i in range(512):
    #     print(str(inputs["input_ids"][i]) + " " + str(inputs["labels"][i]) + " " + str(inputs["attention_mask"][i]))
    # print(inputs["input_ids"])
    # print(inputs["labels"])
    # print(tokenizer.decode(inputs["input_ids"]))
    
    
    # prompt = [tokenizer.get_command("<|system|>")] + tokenizer.encode(sys, add_special_tokens=False)
    # instruction_ = [tokenizer.get_command("<|user|>")] + tokenizer.encode("\n " + "\n".join([q]).strip(), add_special_tokens=False,max_length=512) + [tokenizer.get_command("<|assistant|>")]
    # instruction = tokenizer.encode(prompt + instruction_)
    # response = tokenizer.encode("\n" + a, add_special_tokens=False)
    # input_ids = instruction + response + [tokenizer.eos_token_id]
    # print(tokenizer.decode(input_ids))



