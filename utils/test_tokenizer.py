from chatglm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
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
    tokenizer = ChatGLMTokenizer(vocab_file='chatglm3_tokenizer/tokenizer.model')
    
    sys_text = "你是由wdndev开发的个人助手。"
    user_text = "介绍一下中国。"
    input_txt = "\n".join(["<|system|>", sys_text.strip(), 
                            "<|user|>", user_text.strip(), 
                            "<|assistant|>"]).strip() + "\n"
    
    model_inputs = tokenizer([input_txt], return_tensors="pt")

    print(tokenizer.batch_decode(model_inputs["input_ids"]))

    messages = [
        {"role": "system", "content": "你是由wdndev开发的个人助手。"},
        {"role": "system", "content": "介绍一下中国。"}
    ]
    # print(tokenizer.chat_template)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt")
    print(tokenizer.batch_decode(model_inputs["input_ids"]))


